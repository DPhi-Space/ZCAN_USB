#!/usr/bin/env python3
"""
zcan_usb ping-pong test

End-to-end test for the zcan_usb driver using two SocketCAN interfaces
wired together on the same physical bus (e.g. can0 <-> can1 of the same
NXP USB CANFD DEBUG adapter, TX/RX crossed over).

Supports both classic CAN (8-byte payload) and CAN FD (--fd, up to 64-byte
payload) frames.

Protocol (payload, "length" bytes - 8 for classic, up to 64 for --fd):

    byte 0-1      seq      uint16 BE   frame sequence number
    byte 2        counter  uint8       value incremented on every hop
    byte 3        hop      uint8       0 = ping (sent by --ping-if)
                                        1 = pong (sent by --pong-if)
    byte 4-5      reserved             0x0000
    byte 6..N-3   filler               deterministic pattern (i & 0xFF),
                                        present only when length > 8
    byte N-2..N-1 crc      uint16 BE   CRC-16/CCITT-FALSE over bytes 0..N-3

Flow per sequence number:

    1. ping-if sends a frame with hop=0, counter=seq&0xFF
    2. pong-if receives it, checks the CRC, increments counter,
       sets hop=1, recomputes the CRC and sends it back
    3. ping-if receives the reply, checks CRC/seq/counter/hop and
       records the round-trip time

Requires no third-party packages, only a Linux kernel with SocketCAN
(CAN_RAW) support. Bring the interfaces up first, e.g.:

    sudo ip link set can0 type can bitrate 500000
    sudo ip link set can0 up
    sudo ip link set can1 type can bitrate 500000
    sudo ip link set can1 up

For CAN FD (needs zcan_usb >= 0.8.0, see the driver's known limitation on
dbitrate == bitrate):

    sudo ip link set can0 type can bitrate 500000 dbitrate 500000 fd on
    sudo ip link set can0 up
    sudo ip link set can1 type can bitrate 500000 dbitrate 500000 fd on
    sudo ip link set can1 up

Usage:

    ./pingpong_test.py --ping-if can0 --pong-if can1 -n 1000
    ./pingpong_test.py --ping-if can0 --pong-if can1 -n 1000 --fd --fd-len 64
"""
import argparse
import socket
import statistics
import struct
import sys
import threading
import time

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)
CANFD_FRAME_FMT = "=IBB2x64s"
CANFD_FRAME_SIZE = struct.calcsize(CANFD_FRAME_FMT)
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_EFF_MASK = 0x1FFFFFFF
CANFD_MAX_DLEN = 64
CANFD_BRS = 0x01	# bit-rate switch flag - see driver's known limitation

HOP_PING = 0
HOP_PONG = 1


def crc16_ccitt_false(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_payload(seq: int, counter: int, hop: int, length: int = 8) -> bytes:
    """Build an 8..64 byte payload: 6-byte header + filler + 2-byte CRC."""
    if length < 8 or length > CANFD_MAX_DLEN:
        raise ValueError(f"length must be 8..{CANFD_MAX_DLEN}, got {length}")
    header = struct.pack(">HBB2x", seq & 0xFFFF, counter & 0xFF, hop & 0xFF)
    filler = bytes(i & 0xFF for i in range(length - 8))
    body = header + filler
    crc = crc16_ccitt_false(body)
    return body + struct.pack(">H", crc)


def parse_payload(data: bytes):
    if len(data) < 8:
        return None
    body, crc_recv = data[:-2], struct.unpack(">H", data[-2:])[0]
    crc_calc = crc16_ccitt_false(body)
    seq, counter, hop = struct.unpack(">HBB2x", body[:6])
    filler = body[6:]
    filler_ok = all(b == (i & 0xFF) for i, b in enumerate(filler))
    return {
        "seq": seq,
        "counter": counter,
        "hop": hop,
        "crc_ok": crc_recv == crc_calc,
        "filler_ok": filler_ok,
    }


def open_can_socket(ifname: str, timeout=None, fd: bool = False) -> socket.socket:
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    if fd:
        s.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
    s.bind((ifname,))
    if timeout is not None:
        s.settimeout(timeout)
    return s


def send_frame(sock: socket.socket, can_id: int, data: bytes, extended: bool, fd: bool = False):
    if extended:
        can_id = (can_id & CAN_EFF_MASK) | CAN_EFF_FLAG
    else:
        can_id = can_id & CAN_SFF_MASK
    if fd:
        frame = struct.pack(CANFD_FRAME_FMT, can_id, len(data), 0, data.ljust(CANFD_MAX_DLEN, b"\x00"))
    else:
        frame = struct.pack(CAN_FRAME_FMT, can_id, len(data), data.ljust(8, b"\x00"))
    sock.send(frame)


def recv_frame(sock: socket.socket, fd: bool = False):
    """Read one frame. With fd=True the socket may still deliver a plain
    classic-CAN frame (CAN_MTU bytes) - detect by the actual read size."""
    bufsize = CANFD_FRAME_SIZE if fd else CAN_FRAME_SIZE
    frame = sock.recv(bufsize)
    if len(frame) == CANFD_FRAME_SIZE:
        can_id, dlen, _flags, data = struct.unpack(CANFD_FRAME_FMT, frame)
    elif len(frame) == CAN_FRAME_SIZE:
        can_id, dlen, data = struct.unpack(CAN_FRAME_FMT, frame)
    else:
        return None
    if can_id & CAN_ERR_FLAG:
        return None
    raw_id = can_id & (CAN_EFF_MASK if can_id & CAN_EFF_FLAG else CAN_SFF_MASK)
    return raw_id, data[:dlen]


class Responder(threading.Thread):
    """Runs on --pong-if: verifies ping CRC, increments counter, replies."""

    def __init__(self, sock, id_ping, id_pong, extended, stop_event, verbose, fd=False, length=8):
        super().__init__(daemon=True)
        self.sock = sock
        self.id_ping = id_ping
        self.id_pong = id_pong
        self.extended = extended
        self.stop_event = stop_event
        self.verbose = verbose
        self.fd = fd
        self.length = length
        self.frames_seen = 0
        self.crc_errors = 0
        self.filler_errors = 0
        self.replies_sent = 0
        self.send_errors = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                result = recv_frame(self.sock, self.fd)
            except socket.timeout:
                continue
            except OSError:
                break
            if result is None:
                continue
            can_id, data = result
            if can_id != self.id_ping:
                continue
            self.frames_seen += 1
            parsed = parse_payload(data)
            if parsed is None or not parsed["crc_ok"]:
                self.crc_errors += 1
                if self.verbose:
                    print(f"[pong] CRC error on seq={parsed['seq'] if parsed else '?'}")
                continue
            if not parsed["filler_ok"]:
                self.filler_errors += 1
            if parsed["hop"] != HOP_PING:
                continue
            new_counter = (parsed["counter"] + 1) & 0xFF
            reply = build_payload(parsed["seq"], new_counter, HOP_PONG, self.length)
            try:
                send_frame(self.sock, self.id_pong, reply, self.extended, self.fd)
            except OSError as e:
                self.send_errors += 1
                if self.verbose:
                    print(f"[pong] seq={parsed['seq']} SEND FAILED ({e})")
                continue
            self.replies_sent += 1
            if self.verbose:
                print(f"[pong] seq={parsed['seq']} counter {parsed['counter']}->{new_counter}")


def run_pinger(sock, id_ping, id_pong, extended, count, timeout, interval, verbose, stats,
	       fd=False, length=8):
    latencies = []
    lost = 0
    crc_errors = 0
    mismatches = 0
    filler_errors = 0
    send_errors = 0

    for seq in range(count):
        counter = seq & 0xFF
        payload = build_payload(seq, counter, HOP_PING, length)

        # A leftover near-zero timeout from the previous iteration's recv
        # wait loop (below) still applies to send() on this same socket -
        # reset it first, and treat a blocked/failed send (e.g. because
        # the driver's TX queue is stuck) as a lost frame instead of
        # crashing the whole run.
        sock.settimeout(timeout)
        t_send = time.perf_counter()
        try:
            send_frame(sock, id_ping, payload, extended, fd)
        except OSError as e:
            lost += 1
            send_errors += 1
            if verbose:
                print(f"[ping] seq={seq} SEND FAILED ({e})")
            if interval:
                time.sleep(interval)
            continue

        deadline = t_send + timeout
        got_reply = False
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            sock.settimeout(max(remaining, 0))
            try:
                result = recv_frame(sock, fd)
            except socket.timeout:
                break
            except OSError:
                break
            if result is None:
                continue
            can_id, data = result
            if can_id != id_pong:
                continue
            t_recv = time.perf_counter()
            parsed = parse_payload(data)
            if parsed is None or not parsed["crc_ok"]:
                crc_errors += 1
                continue
            if parsed["seq"] != seq or parsed["hop"] != HOP_PONG:
                continue
            if parsed["counter"] != (counter + 1) & 0xFF:
                mismatches += 1
                continue
            if not parsed["filler_ok"]:
                filler_errors += 1
            latencies.append(t_recv - t_send)
            got_reply = True
            if verbose:
                print(f"[ping] seq={seq} rtt={(t_recv - t_send) * 1000:.3f} ms")
            break

        if not got_reply:
            lost += 1
            if verbose:
                print(f"[ping] seq={seq} LOST (timeout)")

        if interval:
            time.sleep(interval)

    stats["latencies"] = latencies
    stats["lost"] = lost
    stats["crc_errors"] = crc_errors
    stats["mismatches"] = mismatches
    stats["filler_errors"] = filler_errors
    stats["send_errors"] = send_errors


def fmt_ms(x):
    return f"{x * 1000:.3f} ms"


def print_report(args, responder: Responder, stats, wall_time):
    latencies = stats["latencies"]
    sent = args.count
    received = len(latencies)
    lost = stats["lost"]
    success_rate = (received / sent * 100) if sent else 0.0

    print()
    print("=" * 56)
    print(" zcan_usb ping-pong test report")
    print("=" * 56)
    print(f" ping-if           : {args.ping_if}")
    print(f" pong-if           : {args.pong_if}")
    print(f" mode              : {'CAN FD (len=' + str(args.fd_len) + ')' if args.fd else 'classic CAN'}")
    print(f" frames requested  : {sent}")
    print(f" wall time         : {wall_time:.3f} s")
    print("-" * 56)
    print(f" replies received  : {received}")
    print(f" lost (timeout)    : {lost}")
    print(f" send errors       : {stats['send_errors']} (ping side) / {responder.send_errors} (pong side)")
    print(f" seq/counter mism. : {stats['mismatches']}")
    print(f" CRC errors (ping) : {stats['crc_errors']}")
    print(f" CRC errors (pong) : {responder.crc_errors}")
    print(f" filler mismatches : {stats['filler_errors']} (ping side) / {responder.filler_errors} (pong side)")
    print(f" success rate      : {success_rate:.2f} %")
    print("-" * 56)
    if latencies:
        print(f" min RTT           : {fmt_ms(min(latencies))}")
        print(f" avg RTT           : {fmt_ms(statistics.mean(latencies))}")
        print(f" max RTT           : {fmt_ms(max(latencies))}")
        if len(latencies) > 1:
            print(f" stddev RTT        : {fmt_ms(statistics.stdev(latencies))}")
        print(f" throughput        : {received / wall_time:.1f} frames/s")
    else:
        print(" no successful round trips - check wiring / bitrate / interfaces")
    print("=" * 56)


def auto_int(value):
    return int(value, 0)


def main():
    parser = argparse.ArgumentParser(description="zcan_usb can0<->can1 ping-pong test")
    parser.add_argument("--ping-if", default="can0", help="interface that sends the ping (default: can0)")
    parser.add_argument("--pong-if", default="can1", help="interface that replies with the pong (default: can1)")
    parser.add_argument("-n", "--count", type=int, default=100, help="number of frames to send (default: 100)")
    parser.add_argument("--id-ping", type=auto_int, default=0x100, help="CAN ID used for ping frames (default: 0x100)")
    parser.add_argument("--id-pong", type=auto_int, default=0x101, help="CAN ID used for pong frames (default: 0x101)")
    parser.add_argument("--extended", action="store_true", help="use 29-bit extended CAN IDs")
    parser.add_argument("--timeout", type=float, default=0.5, help="per-frame reply timeout in seconds (default: 0.5)")
    parser.add_argument("--interval", type=float, default=0.0, help="delay between pings in seconds (default: 0)")
    parser.add_argument("--fd", action="store_true", help="use CAN FD frames instead of classic CAN (needs zcan_usb >= 0.8.0 and 'fd on')")
    parser.add_argument("--fd-len", type=int, default=32, help="CAN FD payload length in bytes, 8..64 (default: 32, only with --fd)")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every frame")
    args = parser.parse_args()

    if args.fd and not (8 <= args.fd_len <= CANFD_MAX_DLEN):
        print(f"error: --fd-len must be 8..{CANFD_MAX_DLEN}", file=sys.stderr)
        sys.exit(1)
    length = args.fd_len if args.fd else 8

    try:
        ping_sock = open_can_socket(args.ping_if, fd=args.fd)
        pong_sock = open_can_socket(args.pong_if, timeout=0.2, fd=args.fd)
    except OSError as e:
        print(f"error: could not open CAN interface: {e}", file=sys.stderr)
        print("hint: bring the interfaces up first, e.g.:", file=sys.stderr)
        if args.fd:
            print("  sudo ip link set can0 type can bitrate 500000 dbitrate 500000 fd on && sudo ip link set can0 up", file=sys.stderr)
            print("  sudo ip link set can1 type can bitrate 500000 dbitrate 500000 fd on && sudo ip link set can1 up", file=sys.stderr)
        else:
            print("  sudo ip link set can0 type can bitrate 500000 && sudo ip link set can0 up", file=sys.stderr)
            print("  sudo ip link set can1 type can bitrate 500000 && sudo ip link set can1 up", file=sys.stderr)
        sys.exit(1)

    stop_event = threading.Event()
    responder = Responder(pong_sock, args.id_ping, args.id_pong, args.extended, stop_event, args.verbose,
			  fd=args.fd, length=length)
    responder.start()

    time.sleep(0.05)  # let the responder thread settle into its recv loop

    mode_desc = f"CAN FD (len={length})" if args.fd else "classic CAN"
    print(f"Running {args.count} ping-pong cycles ({mode_desc}): {args.ping_if} -> {args.pong_if} -> {args.ping_if}")
    stats = {}
    t0 = time.perf_counter()
    try:
        run_pinger(ping_sock, args.id_ping, args.id_pong, args.extended,
                   args.count, args.timeout, args.interval, args.verbose, stats,
                   fd=args.fd, length=length)
    except KeyboardInterrupt:
        print("\ninterrupted, finishing up...")
        stats.setdefault("latencies", [])
        stats.setdefault("lost", args.count - len(stats.get("latencies", [])))
        stats.setdefault("crc_errors", 0)
        stats.setdefault("mismatches", 0)
        stats.setdefault("filler_errors", 0)
        stats.setdefault("send_errors", 0)
    wall_time = time.perf_counter() - t0

    stop_event.set()
    responder.join(timeout=1.0)
    ping_sock.close()
    pong_sock.close()

    print_report(args, responder, stats, wall_time)


if __name__ == "__main__":
    main()
