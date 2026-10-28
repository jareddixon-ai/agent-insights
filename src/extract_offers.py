"""Step 5-3 (part 2): read offer PDFs, keep the numbers, drop the people.

For each PDF: pull the text out (pdfplumber), work out which form it is, and read the
fields the offer metrics need: price, financing, down payment, contingencies, days to close,
buyer agent, listing, date. Buyer names, phone numbers, emails, notes and signatures are
NEVER stored. Before a row is saved it is checked: if any buyer name, email or phone number
from that PDF appears anywhere in the row, the row is refused.

Rule of thumb (data minimization): store only what a metric uses. Free-text notes are
dropped entirely, because redacting names is not enough: "our kids start at Roosevelt
Elementary in fall" still says who they are.

Writes data/fake/offers_extracted.csv and prints a report of what failed and why.
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

from agent_model import DATA

PDF_DIR = DATA / "offer_pdfs"
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\(?\d{3}\)?[ -]?\d{3}-\d{4}")
FIELDS = ["offer_id", "listing_id", "buyer_agent_id", "offer_date", "offer_price", "financing",
          "down_payment_pct", "inspection_contingency", "appraisal_contingency", "loan_contingency",
          "days_to_close"]


def find(pattern, text):
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"could not find {pattern!r}")
    return m.group(1)


def money(s):
    return float(s.replace(",", "").replace("$", ""))


def parse_form_a(text):
    price = money(find(r"Purchase price: \$([\d,]+)", text))
    financing = find(r"Financing: (.+)", text)
    cash = financing.strip().lower() == "all cash"
    cont = find(r"Contingencies: (.+)", text)
    agent = find(r"Buyer's agent: (\S+)", text)
    return {
        "offer_id": find(r"Offer ID: (O\d+)", text),
        "listing_id": find(r"\(Listing (L\d+)\)", text),
        "buyer_agent_id": "" if agent == "None" else agent,
        "offer_date": find(r"Date: (\d{4}-\d{2}-\d{2})", text),
        "offer_price": price,
        "financing": "cash" if cash else "loan",
        "down_payment_pct": 100 if cash else int(find(r"(\d+)% down", financing)),
        "inspection_contingency": int("Inspection" in cont),
        "appraisal_contingency": int("Appraisal" in cont),
        "loan_contingency": int("Loan" in cont),
        "days_to_close": int(find(r"Close of escrow: (\d+) days", text)),
    }, [find(r"Buyer\(s\): (.+)", text)]


def parse_form_b(text):
    price = money(find(r"Price offered \.+ ([\d,]+) USD", text))
    loan_line = find(r"Loan amount: (.+)", text)
    cash = "none" in loan_line
    flag = lambda name: int(find(rf"{name} contingency: (Yes|Waived)", text) == "Yes")
    agent = re.search(r"Represented by agent (A\d+)", text)
    return {
        "offer_id": find(r"Reference (O\d+)", text),
        "listing_id": find(r"listing (L\d+)", text),
        "buyer_agent_id": agent.group(1) if agent else "",
        "offer_date": find(r"dated (\d{4}-\d{2}-\d{2})", text),
        "offer_price": price,
        "financing": "cash" if cash else "loan",
        "down_payment_pct": 100 if cash else round(100 * (1 - money(find(r"([\d,]+) USD", loan_line)) / price)),
        "inspection_contingency": flag("Inspection"),
        "appraisal_contingency": flag("Appraisal"),
        "loan_contingency": flag("Financing"),
        "days_to_close": int(find(r"Closing: (\d+) days", text)),
    }, [find(r"Buyer name\(s\): (.+)", text)]


def private_bits(text, name_lines):
    """Everything in this PDF that identifies the buyers: each name and name part, emails, phones."""
    names = set()
    for line in name_lines:
        for person in re.split(r";| and ", line):
            person = person.strip()
            if person:
                names.add(person)
                names.update(part for part in person.split() if len(part) > 2)
    return names | set(EMAIL.findall(text)) | set(PHONE.findall(text))


def extract_one(path):
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    if "RESIDENTIAL PURCHASE OFFER" in text:
        row, name_lines = parse_form_a(text)
    elif "OFFER TO PURCHASE REAL PROPERTY" in text:
        row, name_lines = parse_form_b(text)
    else:
        raise ValueError("unknown form")
    # Privacy check before anything is stored: no name, email or phone may survive.
    secret = private_bits(text, name_lines)
    stored = " ".join(str(v) for v in row.values()).lower()
    leaked = [s for s in secret if s.lower() in stored]
    if leaked:
        raise ValueError("refused: private detail would be stored")   # never print what leaked
    return row


def extract_folder(folder=PDF_DIR):
    rows, failures = [], []
    for path in sorted(Path(folder).glob("*.pdf")):
        try:
            rows.append(extract_one(path))
        except Exception as e:   # one bad file must not stop the rest
            failures.append({"file": path.name, "reason": str(e)})
    return pd.DataFrame(rows, columns=FIELDS), pd.DataFrame(failures, columns=["file", "reason"])


def main():
    table, failures = extract_folder()
    print(f"Read {len(table) + len(failures)} offer PDFs: {len(table)} parsed, {len(failures)} failed")
    if len(failures):
        print(failures.to_string(index=False))
    truth = pd.read_csv(DATA / "offers.csv", keep_default_na=False).set_index("offer_id")
    check = truth.loc[table["offer_id"], FIELDS[1:]].reset_index()
    same = (table[FIELDS].astype(str).to_numpy() == check[FIELDS].astype(str).to_numpy()).all(axis=1)
    print(f"Rows that match offers.csv exactly: {same.sum()} of {len(table)}")
    table.to_csv(DATA / "offers_extracted.csv", index=False)
    print("Saved offers_extracted.csv (no names, emails, phones or notes)")


if __name__ == "__main__":
    main()
