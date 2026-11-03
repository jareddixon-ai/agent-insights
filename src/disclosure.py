"""Phase 9: the Disclosure setting. WORDING ONLY.

The tool is for brokers. It shows nothing to agents. Whether a brokerage tells its agents they are scored, and
how, is the brokerage's decision (and a question for its own lawyer: this tool does not give legal advice).
This setting only picks which sentence the readout (src/make_readout.py) and the README's Disclosure section use.
Nothing in scoring, the dashboard numbers or scored/ reads it (a test checks that no other source file imports it).

config/disclosure.json:  "disclosure": "broker_only" (default) or "agents_told"
"""

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "config" / "disclosure.json"
DEFAULT = "broker_only"

WORDING = {
    "broker_only": ("Disclosure: broker only. The brokerage has chosen not to tell its agents they are scored. This tool "
                    "shows nothing to agents. What agents are told is the brokerage's decision, and this tool does not "
                    "give legal advice."),
    "agents_told": ("Disclosure: agents told. The brokerage has chosen to tell its agents that it uses this tool and how. "
                    "The tool itself still shows nothing to agents. What they are told, and when, is the brokerage's "
                    "decision, and this tool does not give legal advice."),
}


def load_setting(path=CONFIG):
    """The chosen setting; anything unknown or missing falls back to broker_only."""
    try:
        value = json.loads(Path(path).read_text()).get("disclosure", DEFAULT)
    except (OSError, ValueError):
        return DEFAULT
    return value if value in WORDING else DEFAULT


def sentence(path=CONFIG):
    return WORDING[load_setting(path)]
