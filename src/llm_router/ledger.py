"""A JSON Lines ledger of every attempt, and a summary of it per model.

    router = load_router("router.toml", on_attempt=JsonlLedger("attempts.jsonl"))
    python -m llm_router.ledger attempts.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

from .base import Attempt


class JsonlLedger:
    """Appends one line per attempt: when, which role and model, how it went, tokens, cost, duration."""

    def __init__(self, path: str):
        self.path = path

    def __call__(self, attempt: Attempt) -> None:
        row = {"ts": round(time.time(), 3), **attempt.as_dict()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(path: str) -> list[dict]:
    """Per provider and model: attempts, successes, failures by kind, known cost, attempts of unknown cost, and
    the average duration of successful calls only (a fast failure would make a slow model look quick)."""
    rows: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "attempts": 0, "ok": 0, "failed": defaultdict(int), "cost": 0.0, "unknown_cost": 0, "ok_ms": 0})
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            a = json.loads(line)
            r = rows[(a["provider"], a["model"])]
            r["attempts"] += 1
            if a["outcome"] == "ok":
                r["ok"] += 1
                r["ok_ms"] += a.get("duration_ms", 0)
            elif a["outcome"] != "retry_more_room":
                r["failed"][a["outcome"]] += 1
            if a.get("cost") is None:
                r["unknown_cost"] += 1
            else:
                r["cost"] += a["cost"]
    out = []
    for (provider, model), r in sorted(rows.items()):
        out.append({
            "provider": provider, "model": model, "attempts": r["attempts"], "ok": r["ok"],
            "failed": dict(r["failed"]), "cost_usd": round(r["cost"], 6), "unknown_cost": r["unknown_cost"],
            "avg_ok_ms": round(r["ok_ms"] / r["ok"]) if r["ok"] else None,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m llm_router.ledger ATTEMPTS.jsonl", file=sys.stderr)
        return 2
    rows = summarize(argv[0])
    print(f"{'provider/model':34} {'tries':>5} {'ok':>4} {'known $':>10} {'unknown':>7} {'avg ok ms':>9}  failures")
    for r in rows:
        name = f"{r['provider']}/{r['model']}"[:34]
        fails = ", ".join(f"{k} {v}" for k, v in sorted(r["failed"].items())) or "-"
        ms = "-" if r["avg_ok_ms"] is None else str(r["avg_ok_ms"])
        print(f"{name:34} {r['attempts']:>5} {r['ok']:>4} {r['cost_usd']:>10.6f} {r['unknown_cost']:>7} {ms:>9}  {fails}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
