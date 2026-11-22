"""Build handbook.pdf from handbook.md, so the notebooks have a real PDF to load.

Run it from the repo root (fpdf2 is only needed for this script):

    uv run --with fpdf2 python 02-rag-fundamentals/data/make_pdf.py
"""

from pathlib import Path

from fpdf import FPDF

HERE = Path(__file__).parent
SOURCE = HERE / "handbook.md"
TARGET = HERE / "handbook.pdf"


def build() -> None:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.set_title("Brightleaf Employee Handbook")
    pdf.add_page()

    for line in SOURCE.read_text().splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 20)
            pdf.multi_cell(0, 10, line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, line[3:], new_x="LMARGIN", new_y="NEXT")
        elif line.strip():
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(3)

    pdf.output(str(TARGET))
    print(f"wrote {TARGET} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    build()
