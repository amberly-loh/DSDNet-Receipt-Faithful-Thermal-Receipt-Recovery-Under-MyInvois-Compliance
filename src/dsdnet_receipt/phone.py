"""Pairing a phone as a receipt scanner for this computer.

The computer shows a QR code holding a one-time pairing code. The phone opens
a small page on the same Wi-Fi network and sends photos with that code; the
computer processes and stores them as usual. Codes are random, kept only in
memory, expire after a set time and can be ended at once from the computer.
"""
from __future__ import annotations

import ipaddress
import secrets
import socket
import threading
import time
from typing import Any

PAIRING_MINUTES = 30
MAX_PHOTOS_PER_PAIRING = 200


class PhonePairings:
    def __init__(self, minutes: int = PAIRING_MINUTES) -> None:
        self.minutes = minutes
        self._lock = threading.Lock()
        self._pairings: dict[str, dict[str, Any]] = {}

    def start(self, user_id: int) -> dict[str, Any]:
        """One active pairing per account: starting again replaces the old code."""
        with self._lock:
            self._expire()
            for token, item in list(self._pairings.items()):
                if item["user_id"] == user_id:
                    del self._pairings[token]
            token = secrets.token_urlsafe(24)
            item = {"user_id": user_id, "expires": time.time() + self.minutes * 60, "received": [], "failed": 0, "busy": 0}
            self._pairings[token] = item
            return {"token": token, **self._public(item)}

    def user_for(self, token: str) -> int | None:
        with self._lock:
            self._expire()
            for known, item in self._pairings.items():
                if secrets.compare_digest(known, token or ""):
                    if len(item["received"]) + item["failed"] >= MAX_PHOTOS_PER_PAIRING:
                        return None
                    return int(item["user_id"])
        return None

    def note(self, token: str, *, receipt: dict[str, Any] | None = None, failed: bool = False, busy: int = 0) -> None:
        with self._lock:
            item = self._pairings.get(token)
            if not item:
                return
            item["busy"] = max(0, item["busy"] + busy)
            if receipt:
                item["received"].append({"public_id": receipt.get("public_id"), "merchant": receipt.get("merchant"),
                                         "total": receipt.get("total_amount"), "date": receipt.get("transaction_date")})
            if failed:
                item["failed"] += 1

    def status(self, user_id: int) -> dict[str, Any] | None:
        with self._lock:
            self._expire()
            for item in self._pairings.values():
                if item["user_id"] == user_id:
                    return self._public(item)
        return None

    def end(self, user_id: int) -> None:
        with self._lock:
            for token, item in list(self._pairings.items()):
                if item["user_id"] == user_id:
                    del self._pairings[token]

    def _expire(self) -> None:
        now = time.time()
        for token, item in list(self._pairings.items()):
            if item["expires"] <= now:
                del self._pairings[token]

    @staticmethod
    def _public(item: dict[str, Any]) -> dict[str, Any]:
        return {"expires_in": max(0, int(item["expires"] - time.time())), "received": list(item["received"]),
                "failed": item["failed"], "processing": item["busy"]}


def lan_addresses() -> list[str]:
    """This computer's addresses on the local network (Wi-Fi / Ethernet),
    most likely first. Nothing is sent: connecting a UDP socket only asks the
    system which interface it would use."""
    found: list[str] = []
    for probe in ("8.8.8.8", "192.168.0.1", "10.255.255.255"):  # default route first
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((probe, 1))
                found.append(sock.getsockname()[0])
        except OSError:
            continue
    try:
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    result = []
    for address in found:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private and not ip.is_loopback and not ip.is_link_local and address not in result:
            result.append(address)
    # the default-route address (normally the Wi-Fi) comes first; virtual
    # adapters (VirtualBox, WSL, VPN) follow as alternatives
    return result


def qr_svg(text: str, *, scale: int = 6, border: int = 4) -> str:
    """The QR code for `text` as an SVG image (error correction level M)."""
    from reportlab.graphics.barcode.qr import QrCodeWidget

    widget = QrCodeWidget(text)
    widget.barLevel = "M"
    widget.qr.make()
    modules = widget.qr.modules
    size = len(modules) + 2 * border
    paths = []
    for row_index, row in enumerate(modules):
        for col_index, dark in enumerate(row):
            if dark:
                paths.append(f"M{col_index + border},{row_index + border}h1v1h-1z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size * scale}" height="{size * scale}" '
            f'shape-rendering="crispEdges" role="img" aria-label="QR code"><rect width="{size}" height="{size}" fill="#fff"/>'
            f'<path fill="#000" d="{"".join(paths)}"/></svg>')
