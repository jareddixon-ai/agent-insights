"""Step 5-3 (part 1): fake offer PDFs, the kind of document a brokerage really has.

Offers arrive as PDFs, not tables. Each fake PDF holds one offer from offers.csv plus the
private details a real one carries: buyer names, phone, email, a note from the buyers, and
signatures. ALL of it is made up: names are random pairs from short lists, phone numbers use
the 555-01xx block reserved for fiction, emails use example.com (reserved for examples).
Two different layouts, because real offers come on different forms.

Writes data/fake/offer_pdfs/<offer_id>.pdf (git-ignored). File names never contain a name.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from agent_model import DATA

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})   # real text, so it can be read back
PDF_DIR = DATA / "offer_pdfs"
N_PDFS = 100
FIRST = ["Marisol", "Theo", "Priya", "Daniel", "Hana", "Lucas", "Imani", "Owen", "Sofia", "Kenji",
         "Ava", "Mateo", "Nora", "Elias", "Leila", "Caleb", "Rosa", "Arjun", "Greta", "Felix"]
LAST = ["Fennick", "Albright", "Castellan", "Dunmore", "Everly", "Galloway", "Hartwell", "Ivers",
        "Kestrel", "Lindqvist", "Marchetti", "Nakamura", "Okafor", "Pemberton", "Quintero", "Rowan"]
NOTES = ["{a} and {b} love the garden and hope to raise our kids here.",
         "We ({a} and {b}) have been looking for two years. This is the one.",
         "{a} grew up nearby and {b} works in town. We would take great care of it."]


def fake_people(rng):
    last = rng.choice(LAST)
    a, b = rng.choice(FIRST, 2, replace=False)
    return f"{a} {last}", f"{b} {last}"


def lines_form_a(o, sale, people, rng):
    a, b = people
    cont = [c for c, flag in [("Inspection (17 days)", o["inspection_contingency"]),
                              ("Appraisal", o["appraisal_contingency"]), ("Loan", o["loan_contingency"])] if flag]
    money = "All cash" if o["financing"] == "cash" else f"Conventional loan, {o['down_payment_pct']}% down"
    return [
        ("FAKE DEMO DOCUMENT - NOT A REAL OFFER", 9, "bold"),
        ("RESIDENTIAL PURCHASE OFFER", 15, "bold"),
        (f"Offer ID: {o['offer_id']}        Date: {o['offer_date']}", 10, "normal"),
        (f"Property: {sale['address']}, Santa Barbara area CA {sale['zip']}  (Listing {o['listing_id']})", 10, "normal"),
        (f"Buyer(s): {a} and {b}", 10, "normal"),
        (f"Buyer phone: (805) 555-01{rng.integers(0, 100):02d}    Email: {a.split()[0].lower()}."
         f"{a.split()[1].lower()}@example.com", 10, "normal"),
        (f"Buyer's agent: {o['buyer_agent_id'] or 'None (buyer unrepresented)'}", 10, "normal"),
        (f"Purchase price: ${o['offer_price']:,.0f}", 12, "bold"),
        (f"Financing: {money}", 10, "normal"),
        (f"Contingencies: {'; '.join(cont) if cont else 'None'}", 10, "normal"),
        (f"Close of escrow: {o['days_to_close']} days after acceptance", 10, "normal"),
        (f"Note from buyers: \"{rng.choice(NOTES).format(a=a.split()[0], b=b.split()[0])}\"", 9, "normal"),
        (f"Signed: {a}      {b}", 10, "normal"),
    ]


def lines_form_b(o, sale, people, rng):
    a, b = people
    yes = lambda f: "Yes" if f else "Waived"
    if o["financing"] == "cash":
        money = "Loan amount: none (all cash)"
    else:
        money = f"Loan amount: {o['offer_price'] * (1 - o['down_payment_pct'] / 100):,.0f} USD"
    return [
        ("FAKE DEMO DOCUMENT - NOT A REAL OFFER", 9, "bold"),
        ("OFFER TO PURCHASE REAL PROPERTY", 14, "bold"),
        (f"Reference {o['offer_id']} / listing {o['listing_id']} / dated {o['offer_date']}", 10, "normal"),
        (f"Buyer name(s): {b}; {a}", 10, "normal"),
        (f"Contact: {b.split()[0].lower()}@example.com, 805-555-01{rng.integers(0, 100):02d}", 10, "normal"),
        (f"Represented by agent {o['buyer_agent_id']}" if o["buyer_agent_id"] else "Buyer is not represented",
         10, "normal"),
        (f"Address: {sale['address']}, {sale['zip']}", 10, "normal"),
        (f"Price offered ........ {o['offer_price']:,.0f} USD", 12, "bold"),
        (money, 10, "normal"),
        (f"Inspection contingency: {yes(o['inspection_contingency'])}", 10, "normal"),
        (f"Appraisal contingency: {yes(o['appraisal_contingency'])}", 10, "normal"),
        (f"Financing contingency: {yes(o['loan_contingency'])}", 10, "normal"),
        (f"Closing: {o['days_to_close']} days", 10, "normal"),
        (f"Buyer signature: {b}", 10, "normal"),
    ]


def write_pdf(path, lines):
    with PdfPages(path, metadata={"Creator": None, "Producer": None, "CreationDate": None}) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        y = 0.94
        for text, size, weight in lines:
            fig.text(0.08, y, text, fontsize=size, fontweight=weight)
            y -= 0.045 if size < 12 else 0.055
        pdf.savefig(fig)
        plt.close(fig)


def make_pdfs(offers, sales, n=N_PDFS, out_dir=PDF_DIR, seed=11):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    pick = offers.iloc[np.sort(rng.choice(len(offers), min(n, len(offers)), replace=False))]
    s = sales.set_index("listing_id")
    for _, o in pick.iterrows():
        people = fake_people(rng)
        form = lines_form_a if rng.random() < 0.5 else lines_form_b
        write_pdf(out_dir / f"{o['offer_id']}.pdf", form(o, s.loc[o["listing_id"]], people, rng))
    return pick["offer_id"].tolist()


def main(n=N_PDFS):
    offers = pd.read_csv(DATA / "offers.csv", keep_default_na=False)
    sales = pd.read_csv(DATA / "sales.csv", keep_default_na=False)
    ids = make_pdfs(offers, sales, n)
    print(f"Wrote {len(ids)} fake offer PDFs to {PDF_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_PDFS)
