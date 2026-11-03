"""Phase 9: fill the owner readout template. Owner version only: the tool shows nothing to agents.

Reads docs/readout_owner_template.md and fills three placeholders:
  {{DISCLOSURE}}  the sentence for the Disclosure setting (config/disclosure.json). Wording only.
  {{SPLIT_HALF}}  the split-half lines from <data_folder>/validation_report.txt, if it exists (counts only).
  {{DATE}}        today's date.
Writes <data_folder>/readout_owner.md in the private data folder (checked git-ignored first). The success-measures
table is left for the owner meeting: every row is a decision about THEIR data and their own use.

Run:  python src/make_readout.py                         (uses config/column_mapping.json)
"""

import argparse
from datetime import date
from pathlib import Path

from disclosure import sentence
from readiness_check import MAPPING, ROOT, UnsafeFolder, check_folder_safe, load_mapping, resolve_folder

TEMPLATE = ROOT / "docs" / "readout_owner_template.md"


def fill(template_text, disclosure, split_half, today=None):
    return (template_text.replace("{{DISCLOSURE}}", disclosure)
            .replace("{{SPLIT_HALF}}", split_half)
            .replace("{{DATE}}", (today or date.today()).isoformat()))


def split_half_lines(folder):
    f = Path(folder) / "validation_report.txt"
    if not f.exists():
        return "Not run yet. Run python src/validate_private.py first."
    keep = [ln for ln in f.read_text().splitlines()
            if ln.startswith(("Sales used", "The two halves agree", "Projected reliability", "Pilot target"))]
    return "\n".join(keep) or "No split-half numbers in the report."


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fill the owner readout template.")
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--data", default=None)
    args = parser.parse_args(argv)
    folder = resolve_folder(load_mapping(args.mapping), args.data)
    try:
        check_folder_safe(folder)
    except UnsafeFolder as e:
        raise SystemExit(str(e))
    out = folder / "readout_owner.md"
    out.write_text(fill(TEMPLATE.read_text(), sentence(), split_half_lines(folder)))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
