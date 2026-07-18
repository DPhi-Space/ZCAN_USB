#!/usr/bin/env python3
"""
zcan_usb ping-pong test

End-to-end test for the zcan_usb driver using two SocketCAN interfaces
wired together on the same physical bus (e.g. can0 <-> can1 of the same
NXP USB CANFD DEBUG adapter, TX/RX crossed over).

Protocol (8-byte classic CAN payload):

    byte 0-1  seq      uint16 BE   frame sequence number
    byte 2    counter  uint8       value incremented on every hop
    byte 3    hop      uint8       0 = ping (sent by --ping-if)
                                    1 = pong (sent by --pong-if)
    byte 4-5  reserved             0x0000
    byte 6-7  crc      uint16 BE   CRC-16/CCITT-FALSE over bytes 0-5

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

Usage:

    ./pingpong_test.py --ping-if can0 --pong-if can1 -n 1000
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
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_EFF_MASK = 0x1FFFFFFF

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


def build_payload(seq: int, counter: int, hop: int) -> bytes:
    body = struct.pack(">HBB2x", seq & 0xFFFF, counter & 0xFF, hop & 0xFF)
    crc = crc16_ccitt_false(body)
    return body + struct.pack(">H", crc)


def parse_payload(data: bytes):
    if len(data) != 8:
        return None
    body, crc_recv = data[:6], struct.unpack(">H", data[6:8])[0]
    crc_calc = crc16_ccitt_false(body)
    seq, counter, hop = struct.unpack(">HBB2x", body)
    return {
        "seq": seq,
        "counter": counter,
        "hop": hop,
        "crc_ok": crc_recv == crc_calc,
    }


def open_can_socket(ifname: str, timeout=None) -> socket.socket:
    s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((ifname,))
    if timeout is not None:
        s.settimeout(timeout)
    return s


def send_frame(sock: socket.socket, can_id: int, data: bytes, extended: bool):
    if extended:
        can_id = (can_id & CAN_EFF_MASK) | CAN_EFF_FLAG
    else:
        can_id = can_id & CAN_SFF_MASK
    frame = struct.pack(CAN_FRAME_FMT, can_id, len(data), data.ljust(8, b"\x00"))
    sock.send(frame)


def recv_frame(sock: socket.socket):
    frame = sock.recv(CAN_FRAME_SIZE)
    can_id, can_dlc, data = struct.unpack(CAN_FRAME_FMT, frame)
    if can_id & CAN_ERR_FLAG:
        return None
    raw_id = can_id & (CAN_EFF_MASK if can_id & CAN_EFF_FLAG else CAN_SFF_MASK)
    return raw_id, data[:can_dlc]


class Responder(threading.Thread):
    """Runs on --pong-if: verifies ping CRC, increments counter, replies."""

    def __init__(self, sock, id_ping, id_pong, extended, stop_event, verbose):
        super().__init__(daemon=True)
        self.sock = sock
        self.id_ping = id_ping
        self.id_pong = id_pong
        self.extended = extended
        self.stop_event = stop_event
        self.verbose = verbose
        self.frames_seen = 0
        self.crc_errors = 0
        self.replies_sent = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                result = recv_frame(self.sock)
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
            if parsed["hop"] != HOP_PING:
                continue
            new_counter = (parsed["counter"] + 1) & 0xFF
            reply = build_payload(parsed["seq"], new_counter, HOP_PONG)
            send_frame(self.sock, self.id_pong, reply, self.extended)
            self.replies_sent += 1
            if self.verbose:
                print(f"[pong] seq={parsed['seq']} counter {parsed['counter']}->{new_counter}")


def run_pinger(sock, id_ping, id_pong, extended, count, timeout, interval, verbose, stats):
    latencies = []
    lost = 0
    crc_errors = 0
    mismatches = 0

    for seq in range(count):
        counter = seq & 0xFF
        payload = build_payload(seq, counter, HOP_PING)
        t_send = time.perf_counter()
        send_frame(sock, id_ping, payload, extended)

        deadline = t_send + timeout
        got_reply = False
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            sock.settimeout(max(remaining, 0))
            try:
                result = recv_frame(sock)
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
    print(f" frames requested  : {sent}")
    print(f" wall time         : {wall_time:.3f} s")
    print("-" * 56)
    print(f" replies received  : {received}")
    print(f" lost (timeout)    : {lost}")
    print(f" seq/counter mism. : {stats['mismatches']}")
    print(f" CRC errors (ping) : {stats['crc_errors']}")
    print(f" CRC errors (pong) : {responder.crc_errors}")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="print every frame")
    args = parser.parse_args()

    try:
        ping_sock = open_can_socket(args.ping_if)
        pong_sock = open_can_socket(args.pong_if, timeout=0.2)
    except OSError as e:
        print(f"error: could not open CAN interface: {e}", file=sys.stderr)
        print("hint: bring the interfaces up first, e.g.:", file=sys.stderr)
        print("  sudo ip link set can0 type can bitrate 500000 && sudo ip link set can0 up", file=sys.stderr)
        print("  sudo ip link set can1 type can bitrate 500000 && sudo ip link set can1 up", file=sys.stderr)
        sys.exit(1)

    stop_event = threading.Event()
    responder = Responder(pong_sock, args.id_ping, args.id_pong, args.extended, stop_event, args.verbose)
    responder.start()

    time.sleep(0.05)  # let the responder thread settle into its recv loop

    print(f"Running {args.count} ping-pong cycles: {args.ping_if} -> {args.pong_if} -> {args.ping_if}")
    stats = {}
    t0 = time.perf_counter()
    try:
        run_pinger(ping_sock, args.id_ping, args.id_pong, args.extended,
                   args.count, args.timeout, args.interval, args.verbose, stats)
    except KeyboardInterrupt:
        print("\ninterrupted, finishing up...")
        stats.setdefault("latencies", [])
        stats.setdefault("lost", args.count - len(stats.get("latencies", [])))
        stats.setdefault("crc_errors", 0)
        stats.setdefault("mismatches", 0)
    wall_time = time.perf_counter() - t0

    stop_event.set()
    responder.join(timeout=1.0)
    ping_sock.close()
    pong_sock.close()

    print_report(args, responder, stats, wall_time)


if __name__ == "__main__":
    main()
