# -*- coding: utf-8 -*-
"""Build a clean printable PDF of Daemonologie (1597 first edition text)
from the Project Gutenberg transcription (#25929)."""

import re
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                PageBreak)

import sys
MODERN = len(sys.argv) > 1 and sys.argv[1] == "modern"
if MODERN:
    SRC = "daemonologie_modern.txt"
    OUT = "Daemonologie_1597_Modern.pdf"
else:
    SRC = "daemonologie_gutenberg.txt"
    OUT = "Daemonologie_1597.pdf"
FONT_DIR = "C:/Windows/Fonts/"

pdfmetrics.registerFont(TTFont("Palatino", FONT_DIR + "pala.ttf"))
pdfmetrics.registerFont(TTFont("Palatino-Italic", FONT_DIR + "palai.ttf"))
pdfmetrics.registerFont(TTFont("Palatino-Bold", FONT_DIR + "palab.ttf"))
pdfmetrics.registerFont(TTFont("Palatino-BoldItalic", FONT_DIR + "palabi.ttf"))
registerFontFamily("Palatino", normal="Palatino", bold="Palatino-Bold",
                   italic="Palatino-Italic", boldItalic="Palatino-BoldItalic")

INK = colors.HexColor("#1a1a1a")

def st(name, **kw):
    base = dict(fontName="Palatino", fontSize=11, leading=15.5,
                textColor=INK, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)

S_BODY     = st("body", firstLineIndent=18, spaceAfter=2)
S_NOIND    = st("noind", spaceAfter=6)
S_ARG      = st("argument", fontName="Palatino-Italic", fontSize=10.5,
                leading=14, leftIndent=36, rightIndent=36, spaceAfter=10)
S_PART     = st("part", fontSize=20, leading=26, alignment=TA_CENTER,
                spaceAfter=14, spaceBefore=6)
S_SUBPART  = st("subpart", fontSize=13, leading=17, alignment=TA_CENTER,
                fontName="Palatino-Italic", spaceAfter=20)
S_CHAP     = st("chap", fontSize=14, leading=18, alignment=TA_CENTER,
                spaceBefore=16, spaceAfter=6)
S_LABEL    = st("label", fontSize=9.5, leading=12, alignment=TA_CENTER,
                spaceAfter=6)
S_ARGTXT   = st("arglabel", fontSize=10, leading=13, alignment=TA_CENTER,
                spaceBefore=2, spaceAfter=4)
S_SCENE    = st("scene", fontName="Palatino-Italic", fontSize=10.5,
                alignment=TA_CENTER, spaceBefore=4, spaceAfter=12)
S_SPEAKER  = st("speaker", fontSize=10.5, leading=13, alignment=TA_CENTER,
                fontName="Palatino-Bold", spaceBefore=8, spaceAfter=4)
S_TITLE1   = st("t1", fontSize=30, leading=38, alignment=TA_CENTER)
S_TITLE2   = st("t2", fontSize=14, leading=20, alignment=TA_CENTER)
S_TITLE3   = st("t3", fontSize=12, leading=18, alignment=TA_CENTER)
S_FINIS    = st("finis", fontSize=12, leading=16, alignment=TA_CENTER,
                spaceBefore=18)
S_TOC_H    = st("toch", fontSize=13, leading=16, alignment=TA_LEFT,
                fontName="Palatino-Bold", spaceBefore=8, spaceAfter=2)
S_TOC_I    = st("toci", fontSize=11, leading=16, alignment=TA_LEFT,
                leftIndent=24)
S_TITLEBLK = st("titleblk", fontSize=11.5, leading=16, alignment=TA_CENTER,
                spaceAfter=8)
S_SUBTLE   = st("subtitle", fontName="Palatino-Italic", fontSize=11,
                leading=15, alignment=TA_CENTER, spaceAfter=8)

# ---------- read and slice ----------

raw = open(SRC, encoding="utf-8").read()
if "*** START OF THE PROJECT GUTENBERG EBOOK" in raw:
    start = raw.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = raw.index("\n", start) + 1
    end = raw.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    text = raw[start:end]
else:
    text = raw

# fix the one known transcription glitch
text = text.replace("to haue bene practised]", "to haue bene practised)")

# drop illustration placeholder lines
text = re.sub(r"^\s*\[Illustration:[^\]]*\]\s*$", "", text, flags=re.M)

lines = [ln.rstrip() for ln in text.split("\n")]
while lines and not lines[0].strip():
    lines.pop(0)

def blocks(ls):
    out, cur = [], []
    for ln in ls:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            out.append(" ".join(cur)); cur = []
    if cur:
        out.append(" ".join(cur))
    return out

# title page = everything before CONTENTS
i = next(k for k, ln in enumerate(lines) if ln.strip() == "CONTENTS")
title_lines = [ln.strip() for ln in lines[:i] if ln.strip()]
rest = lines[i:]

# contents = blocks between CONTENTS and THE PREFACE
j = next(k for k, ln in enumerate(rest) if "PREFACE" in ln)
contents_lines = [ln.rstrip() for ln in rest[1:j] if ln.strip()]
body_lines = rest[j:]

# ---------- markup helpers ----------

def markup(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"_([^_]+)_", r"<i>\1</i>", s)
    s = s.replace("_", "")
    return s

SPEAKER_RE = re.compile(
    r"^(_?(?:PHI|Phi|EPI|Epi|PHILOMATHES|EPISTEMON)\.?_?)\.?\s+(.*)$",
    re.S)

def body_para(s, style=S_BODY):
    m = SPEAKER_RE.match(s)
    if m:
        sp = m.group(1).strip("_").upper()
        name = {"PHI": "Phi.", "PHILOMATHES": "Philomathes.",
                "EPI": "Epi.", "EPISTEMON": "Epistemon."}[sp.rstrip(".")]
        return Paragraph("<b>%s</b>&nbsp;&nbsp;%s" % (name, markup(m.group(2))),
                         style)
    return Paragraph(markup(s), style)

# ---------- build story ----------

story = []

# --- title page ---
story.append(Spacer(1, 1.6 * inch))
story.append(Paragraph("Daemonologie", S_TITLE1))
story.append(Spacer(1, 14))
if MODERN:
    story.append(Paragraph("In Form of a Dialogue,", S_TITLE2))
    story.append(Paragraph("Divided into three Books.", S_TITLE2))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>A modern-spelling edition</i>", S_TITLE3))
else:
    story.append(Paragraph("In Forme of a Dialogie,", S_TITLE2))
    story.append(Paragraph("Diuided into three Bookes.", S_TITLE2))
story.append(Spacer(1, 26))
story.append(Paragraph("By James R<super>X</super>", S_TITLE2))
story.append(Spacer(1, 90))
if MODERN:
    story.append(Paragraph("Printed by Robert Waldegrave,<br/>"
                           "Printer to the King's Majesty. An. 1597.",
                           S_TITLE3))
else:
    story.append(Paragraph("Printed by Robert Walde-graue,<br/>"
                           "Printer to the Kings Majestie. An. 1597.",
                           S_TITLE3))
story.append(Spacer(1, 8))
story.append(Paragraph("<i>Cum Privilegio Regio.</i>", S_TITLE3))
story.append(PageBreak())

# --- contents ---
story.append(Spacer(1, 0.4 * inch))
story.append(Paragraph("CONTENTS", S_PART))
story.append(Spacer(1, 16))
for ln in contents_lines:
    indent = len(ln) - len(ln.lstrip())
    txt = ln.strip()
    if indent:
        story.append(Paragraph(txt, S_TOC_I))
    else:
        story.append(Paragraph(txt, S_TOC_H))
story.append(PageBreak())

# --- body ---
blks = blocks(body_lines)
k = 0
pending_speaker = None
expect_argument = False
newes_title = False

while k < len(blks):
    b = blks[k]
    t = b.strip()

    if t in ("FIRST BOOKE.", "SECONDE BOOKE.", "THIRDE BOOKE.",
             "FIRST BOOK.", "SECOND BOOK.", "THIRD BOOK."):
        story.append(PageBreak())
        story.append(Spacer(1, 1.2 * inch))
        story.append(Paragraph(t, S_PART))
        k += 1
        continue

    if t in ("NEWES FROM SCOTLAND.", "NEWS FROM SCOTLAND."):
        story.append(PageBreak())
        story.append(Spacer(1, 0.8 * inch))
        story.append(Paragraph(t, S_PART))
        newes_title = True
        k += 1
        continue

    if newes_title:
        if t == "To the Reader.":
            newes_title = False
            story.append(PageBreak())
            story.append(Spacer(1, 0.5 * inch))
            story.append(Paragraph("TO THE READER.", S_PART))
        else:
            story.append(Paragraph(markup(t), S_TITLEBLK))
        k += 1
        continue

    if t == "Discourse.":
        story.append(Paragraph("DISCOURSE.", S_PART))
        # next two blocks are the pamphlet subtitle
        for _x in range(2):
            k += 1
            story.append(Paragraph(markup(blks[k]), S_SUBTLE))
        story.append(Spacer(1, 10))
        k += 1
        continue

    if t == "THE PREFACE. TO THE READER.":
        story.append(Paragraph("THE PREFACE.", S_PART))
        story.append(Paragraph("<i>To the Reader.</i>", S_SUBPART))
        k += 1
        continue

    if re.match(r"^Chap\.", t):
        label = re.sub(r"^Chap\.\s*", "CHAPTER " if MODERN else "CHAP. ", t)
        story.append(Paragraph(label.upper(), S_CHAP))
        k += 1
        continue

    if t in ("ARGVMENT.", "ARGUMENT."):
        lbl = "A R G U M E N T ." if MODERN else "A R G V M E N T ."
        story.append(Paragraph(lbl, S_ARGTXT))
        expect_argument = True
        k += 1
        continue

    if expect_argument:
        story.append(Paragraph(markup(t), S_ARG))
        expect_argument = False
        k += 1
        continue

    if t == "PHILOMATHES and EPISTEMON reason the matter.":
        story.append(Paragraph("<i>%s</i>" % t, S_SCENE))
        k += 1
        continue

    if t in ("PHILOMATHES.", "EPISTEMON."):
        pending_speaker = t.rstrip(".")
        k += 1
        continue

    if t in ("FINIS.", "_FINIS._"):
        story.append(Paragraph("F I N I S .", S_FINIS))
        k += 1
        continue

    if pending_speaker:
        b2 = pending_speaker + ". " + t
        pending_speaker = None
        story.append(body_para(b2, S_NOIND))
    else:
        story.append(body_para(t))
    k += 1

# ---------- doc ----------

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Palatino", 9)
    canvas.setFillColor(INK)
    canvas.drawCentredString(LETTER[0] / 2.0, 0.55 * inch,
                             str(canvas.getPageNumber()))
    canvas.restoreState()

doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=1.1 * inch, rightMargin=1.1 * inch,
                        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                        title="Daemonologie (1597)",
                        author="King James I")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
print("wrote", OUT)
