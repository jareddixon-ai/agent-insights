# Deletion checklist and letter

Run `python src/delete_private.py` first. It lists every file it would delete (a dry run). With `--confirm` it deletes the private data folder and writes a log of file names and sizes, never contents. Then work through the hand checklist, because code can't reach these.

## Hand checklist
- [ ] Backups: Time Machine, other backup drives and cloud backups.
- [ ] Cloud sync: iCloud Drive, Dropbox, Google Drive or OneDrive copies.
- [ ] Email and chat: any export or report sent by email, Slack or text, and the downloads folder.
- [ ] Other machines: a second computer or USB drive the files were copied to.
- [ ] The Trash or Recycle Bin, emptied.
- [ ] Broker notes (`context_notes.csv`) are in the private folder and go with it. Copy any the broker wants to keep first.
- [ ] The key that links agent codes to names stays with the brokerage. It was never given to the builder.

Deleting a file on a modern drive does not guarantee the bytes are gone. Full-disk encryption, turned on before any data arrived, keeps deleted data unreadable.

## Letter
Date: ____________

To: ____________ (brokerage contact)
From: ____________ (builder)

I confirm that on ____________ I deleted all data the brokerage provided for the agent-insights pilot, and everything derived from it, from every machine and account I control. This includes the export files, the cleaned tables, the scores, the sale lists, the validation report, the readout and broker notes, and any backups I made.

The deletion log from `src/delete_private.py` (file names and sizes only) is attached. The brokerage's contact watched the deletion.

I kept: the open-source code, and results on fake data only.

Signed: ____________ (builder)    Date: ____________
Witnessed: ____________ (brokerage contact)    Date: ____________
