"""Render the private abstract draft; never publish or claim quality acceptance."""
from html import escape
import argparse
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def ascii_text(text):
    return text.translate(str.maketrans({'–': '-', '—': '-', '’': "'",
                                        '“': '"', '”': '"', '→': 'to'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/pdf/one-beyond-technical-abstract-draft.pdf')
    args = parser.parse_args()
    source = ROOT / 'reports/submission-prep-oct23/technical-abstract-draft.md'
    text = ascii_text(source.read_text(encoding='utf-8'))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='DraftTitle', fontName='Helvetica-Bold', fontSize=21,
                              leading=25, textColor=colors.HexColor('#172b35'), spaceAfter=12))
    styles.add(ParagraphStyle(name='DraftBody', fontName='Helvetica', fontSize=10,
                              leading=14.5, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name='DraftHeading', fontName='Helvetica-Bold', fontSize=12,
                              leading=16, spaceBefore=10, spaceAfter=6, keepWithNext=True,
                              textColor=colors.HexColor('#173f4c')))
    story = []
    for block in re.split(r'\n\s*\n', text.strip()):
        line = ' '.join(block.splitlines())
        if line.startswith('# '):
            story.append(Paragraph(escape(line[2:]), styles['DraftTitle']))
        elif line.startswith('## '):
            if line.startswith('## 4.'):
                story.append(PageBreak())
            story.append(Paragraph(escape(line[3:]), styles['DraftHeading']))
        else:
            story.append(Paragraph(escape(line), styles['DraftBody']))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph('Private draft | 23 October 2026 | Not submitted or quality-certified',
                           styles['DraftBody']))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d0d9dd'))
        canvas.line(19 * mm, 17 * mm, A4[0] - 19 * mm, 17 * mm)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#52626b'))
        canvas.drawString(19 * mm, 12 * mm, 'ONE BEYOND / TECHNICAL ABSTRACT / PRIVATE DRAFT')
        canvas.drawRightString(A4[0] - 19 * mm, 12 * mm, f'{doc.page}')
        canvas.restoreState()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=19 * mm,
                            leftMargin=19 * mm, topMargin=18 * mm, bottomMargin=23 * mm,
                            title='One Beyond - Technical Abstract (Private Draft)',
                            author='One Beyond')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(args.output)
    extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
    if not 1 <= len(reader.pages) <= 5:
        raise ValueError('Abstract exceeds the five-page limit')
    for heading in ['1. Objective', '2. Implemented', '3. Engineering',
                    '4. Evidence', '5. Limitations']:
        if heading not in extracted:
            raise ValueError(f'Missing section: {heading}')
    print(f'Created {args.output}; pages={len(reader.pages)}; draft only; render inspection required')


if __name__ == '__main__':
    main()
