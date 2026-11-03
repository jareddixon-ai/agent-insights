"""Phase 9: end of a pilot. Delete a private data folder, listing every file it removes.

The pilot promise: when the pilot ends (or is stopped early), everything the brokerage gave us is deleted
within 5 business days, with the brokerage's staff watching, and a signed letter says so. This script is the
"watching" part: it shows exactly what it will delete, deletes only after --confirm, and writes a log of
every file it removed (path and size, never contents).

It deletes the whole private data folder: the brokerage's export copies, standard/, scored/, validation_report.txt
and the broker-notes file (tags only) if it sits in the folder. Safety rules (each one is tested):
  - Dry run by default. Nothing is deleted without --confirm.
  - Refuses the repo folder itself, a folder that contains the repo, a home folder, a drive root, or any folder
    that holds a .git folder.
  - Refuses to touch a file git tracks (a private folder must never hold tracked files).
  - The log is written OUTSIDE the folder being deleted, and the log holds file names and sizes only.
  - Symbolic links are removed as links; what they point to is never followed or deleted.

What code can't do is listed at the end of every run as a hand checklist: copies in backups, cloud sync, email,
other machines. On a modern drive deleting a file does not guarantee the bytes are gone; turn on full-disk
encryption (FileVault on a Mac) before any private data arrives, so deleted data stays unreadable.

Run:  python src/delete_private.py                         (dry run: lists everything it would delete)
      python src/delete_private.py --confirm               (deletes, then writes deletion_log_<date>.txt)
      python src/delete_private.py --data /path/to/folder --log /path/outside/folder/log.txt --confirm
"""

import argparse
import shutil
import subprocess
from datetime import date
from pathlib import Path

from readiness_check import MAPPING, ROOT, load_mapping, resolve_folder

HAND_CHECKLIST = [
    "Backups: Time Machine, other backup drives and cloud backups (the folder may be in an old snapshot).",
    "Cloud sync: iCloud Drive, Dropbox, Google Drive or OneDrive copies of the folder or its files.",
    "Email and chat: any export or report sent by email, Slack or text, and the downloads folder.",
    "Other machines: any second computer or USB drive the export or reports were copied to.",
    "The Trash or Recycle Bin, then empty it.",
    "The broker-notes file (tags only) if it was kept outside this folder; the brokerage decides whether to keep its own copy first.",
    "Anything the brokerage's own team exported to share with us (their records, not ours to keep).",
    "Sign the deletion letter (docs/deletion_letter_template.md) with the brokerage's contact, while they watch the folder go.",
]


class UnsafeDelete(Exception):
    """The folder is not one this script may delete."""


def tracked_files(folder, repo=ROOT):
    """Files in the folder that git tracks, if the folder is inside a git repo (none expected)."""
    r = subprocess.run(["git", "-C", str(folder), "ls-files", "--full-name", str(folder)],
                       capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def check_safe_to_delete(folder, log, repo=ROOT):
    folder, repo, home = Path(folder).resolve(), Path(repo).resolve(), Path.home().resolve()
    if not folder.exists() or not folder.is_dir():
        raise UnsafeDelete(f"'{folder.name}' is not a folder that exists. Nothing to delete.")
    if folder == repo or folder in repo.parents:
        raise UnsafeDelete("Refusing: that folder is the repo or contains the repo. Point --data at the private data folder only.")
    if folder == home or folder in home.parents or folder == Path(folder.anchor):
        raise UnsafeDelete("Refusing: that is a home folder or a drive root.")
    if (folder / ".git").exists():
        raise UnsafeDelete("Refusing: the folder holds a .git folder, so it is a repo, not a private data folder.")
    if tracked_files(folder):
        raise UnsafeDelete("Refusing: git tracks files in this folder. A private data folder must never hold tracked files.")
    log = Path(log).resolve()
    if log == folder or folder in log.parents:
        raise UnsafeDelete("Refusing: the log would be written inside the folder being deleted. Put it somewhere else.")
    return folder


def list_files(folder):
    """Every file and link under the folder: (relative path, size in bytes). Links count as links."""
    rows = []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_symlink() or p.is_file():
            rows.append((str(p.relative_to(folder)), p.lstat().st_size))
    return rows


def delete_folder(folder, log, confirm=False, repo=ROOT):
    """Dry run unless confirm. Returns the list of (path, size) it listed (and, with confirm, removed)."""
    folder = check_safe_to_delete(folder, log, repo)
    rows = list_files(folder)
    if confirm:
        shutil.rmtree(folder)            # rmtree removes symlinks as links and never follows them
        lines = [f"Deletion log, {date.today().isoformat()}", f"Folder deleted: {folder.name}",
                 f"Files removed: {len(rows)}, total {sum(s for _, s in rows):,} bytes", "(file names and sizes only; no contents)", ""]
        lines += [f"{size:>12,}  {path}" for path, size in rows]
        Path(log).write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Delete a private data folder and list every file removed.")
    parser.add_argument("--mapping", default=str(MAPPING))
    parser.add_argument("--data", default=None, help="data folder (overrides data_folder in the mapping)")
    parser.add_argument("--log", default=f"deletion_log_{date.today().isoformat()}.txt")
    parser.add_argument("--confirm", action="store_true", help="actually delete (without it, this is a dry run)")
    args = parser.parse_args(argv)
    folder = resolve_folder(load_mapping(args.mapping), args.data)
    try:
        rows = delete_folder(folder, args.log, args.confirm)
    except UnsafeDelete as e:
        raise SystemExit(str(e))
    print(("DELETED" if args.confirm else "DRY RUN: would delete") + f" {len(rows)} files from '{Path(folder).name}':")
    for path, size in rows:
        print(f"  {size:>12,}  {path}")
    if args.confirm:
        print(f"\nLog written to {Path(args.log).resolve()} (names and sizes only).")
    else:
        print("\nNothing was deleted. Run again with --confirm while the brokerage's contact watches.")
    print("\nBy hand (code can't reach these):")
    for item in HAND_CHECKLIST:
        print(f"  [ ] {item}")


if __name__ == "__main__":
    main()
