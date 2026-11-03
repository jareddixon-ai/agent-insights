"""Phase 9: broker notes on the coaching brief. Tags, not free text.

A broker can add, for one agent: a specialty tag, a client goal tag on a sale (price, speed or certainty),
and a "this sale looks wrong" flag on a sale. Why tags and not a text box: a box invites client names, health
news and gossip into a file. A fixed list can't hold any of that.

Where they go: ONE local file, data/private/context_notes.csv (the folder is git-ignored), or the path in the
AGENT_INSIGHTS_NOTES environment variable. Never the repo, never the scores.
What they do to the numbers: nothing. No score, range or call reads this file (a test checks that no scoring
file imports it). The brief only uses the tags to label sales and to list sales with a speed or certainty
goal separately, since a client who wanted a fast or sure sale gave up price on purpose.
Privacy: this code never prints or logs a tag. The brief shows tags back to the broker who is running the app,
on that machine, and nowhere else. The public demo turns the box off, so nothing typed into the hosted demo
is stored. Fake data only in this repo.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECIALTIES = ["Condos", "Estates and probate", "New construction", "Luxury", "Investment properties",
               "Relocation", "First-time buyers"]
GOALS = ["Price", "Speed", "Certainty"]            # what the client cared about most on one sale
FIELDS = ["saved_at", "agent_id", "sale_id", "kind", "value"]
KINDS = {"specialty": SPECIALTIES, "goal": GOALS, "looks_wrong": ["yes", "no"]}
NEEDS_SALE = {"goal", "looks_wrong"}


def notes_path(env=None, root=ROOT):
    env = os.environ if env is None else env
    if env.get("AGENT_INSIGHTS_NOTES"):
        return Path(env["AGENT_INSIGHTS_NOTES"])
    return Path(root) / "data" / "private" / "context_notes.csv"


def save_tag(agent_id, kind, value, sale_id="", path=None):
    """Append one tag. Returns True if saved, False if the kind or value is not on the fixed list or a sale is missing."""
    if kind not in KINDS or value not in KINDS[kind] or (kind in NEEDS_SALE and not sale_id):
        return False
    path = Path(path) if path else notes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), agent_id, sale_id if kind in NEEDS_SALE else "",
                    kind, value])
    try:
        os.chmod(path, 0o600)   # only the machine's user can read it
    except OSError:
        pass
    return True


def read_tags(agent_id, path=None):
    """This agent's current tags, for the brief on the broker's own screen:
    {'specialties': [..], 'goals': {sale_id: goal}, 'flags': {sale_ids}}. The latest entry wins."""
    out = {"specialties": [], "goals": {}, "flags": set()}
    path = Path(path) if path else notes_path()
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("agent_id") != agent_id:
                continue
            kind, value, sale = row.get("kind"), row.get("value"), row.get("sale_id")
            if kind == "specialty" and value not in out["specialties"]:
                out["specialties"].append(value)
            elif kind == "goal":
                out["goals"][sale] = value
            elif kind == "looks_wrong":
                (out["flags"].add if value == "yes" else out["flags"].discard)(sale)
    return out


def count_tags(agent_id, path=None):
    """How many tags exist for this agent (a number, never the tags)."""
    t = read_tags(agent_id, path)
    return len(t["specialties"]) + len(t["goals"]) + len(t["flags"])
