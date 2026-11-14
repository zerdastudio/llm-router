"""A spending allowance per role, over a time window, with reservations for calls in flight."""
from __future__ import annotations

import math
import time

from .base import BudgetExhausted
from .limits import Clock


class Reservation:
    """The worst-case cost of one call, held against the cap until the call ends."""

    __slots__ = ("amount", "open")

    def __init__(self, amount: float):
        self.amount = amount
        self.open = True


class RoleBudget:
    """`cap_usd` per window of `window_s` seconds (None: one window that never reopens).

    Every call reserves its worst-case cost before it is sent, and is sent only if that fits:
    spent + held by calls in flight + this call's worst case <= cap. When the answer comes back the reservation
    becomes the real cost; when the call fails or is cancelled, the reservation is released.

    So the cap holds under any number of concurrent requests, as long as each worst case is not below the real
    cost. The output side of the worst case is exact (the whole output room at the highest output price); the
    input side is an estimate from characters, so a provider that counts far more input tokens than expected can
    still take the spending a little past the cap.

    The window starts with the first call reserved in it. No method awaits, so concurrent tasks on one event loop
    cannot interleave inside them.
    """

    def __init__(self, cap_usd: float, window_s: float | None = 86400.0, clock: Clock = time.monotonic):
        if not (isinstance(cap_usd, (int, float)) and not isinstance(cap_usd, bool) and math.isfinite(cap_usd)
                and cap_usd > 0):
            raise ValueError(f"the cap must be a positive number of dollars, not {cap_usd!r}")
        if window_s is not None and not (math.isfinite(window_s) and window_s > 0):
            raise ValueError(f"the window must be a positive number of seconds or None, not {window_s!r}")
        self.cap = float(cap_usd)
        self.window_s = window_s
        self.spent = 0.0
        self.reserved = 0.0
        self._start: float | None = None
        self._clock = clock

    def _roll(self) -> None:
        if self.window_s is not None and self._start is not None and self._clock() - self._start >= self.window_s:
            self.spent = 0.0
            self._start = None   # calls still in flight keep their reservations, and settle into the new window

    def resets_in(self) -> float | None:
        self._roll()
        if self.window_s is None or self._start is None:
            return None
        return max(self._start + self.window_s - self._clock(), 0.0)

    def is_open(self) -> bool:
        self._roll()
        return self.spent + self.reserved < self.cap

    def reserve(self, role: str, amount: float, attempts: list | None = None) -> Reservation:
        """Hold `amount` for one call, or raise BudgetExhausted if it does not fit."""
        self._roll()
        amount = max(float(amount), 0.0)
        if self.spent + self.reserved + amount > self.cap:
            raise BudgetExhausted(role, self.spent, self.cap, self.resets_in(), needed=amount,
                                  reserved=self.reserved, attempts=attempts)
        if self._start is None:
            self._start = self._clock()
        self.reserved += amount
        return Reservation(amount)

    def release(self, reservation: Reservation) -> None:
        if reservation.open:
            reservation.open = False
            self.reserved = max(self.reserved - reservation.amount, 0.0)

    def settle(self, reservation: Reservation, cost: float) -> None:
        """The call came back: free its reservation and charge what it really cost."""
        self.release(reservation)
        self._roll()
        self.add(cost)

    def add(self, cost: float) -> None:
        if cost and isinstance(cost, (int, float)) and math.isfinite(cost) and cost > 0:
            self.spent += cost
