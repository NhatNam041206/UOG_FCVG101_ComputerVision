"""Conference-paper styling for the Word version of the report (used by build_report.py).

make_reference(path)            writes a pandoc reference.docx: A4, Times 10 pt, black, IEEE-like headings and captions
finish(docx_path, wide_images)  two-column body with single-column title block, wide figures and wide tables,
                                three-rule tables, column-width images, page numbers
"""
import copy
import subprocess

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
PAGE_W, PAGE_H = Cm(21.0), Cm(29.7)
MARGIN_TB, MARGIN_LR = Cm(1.9), Cm(1.5)
GAP = Cm(0.6)
TEXT_W = PAGE_W - 2 * MARGIN_LR
COL_W = (TEXT_W - GAP) // 2
BLACK = RGBColor(0, 0, 0)


def _font(style, size, bold=False, italic=False, small_caps=False):
    f = style.font
    f.name, f.size, f.bold, f.italic, f.small_caps = FONT, Pt(size), bold, italic, small_caps
    f.color.rgb = BLACK
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        fonts.attrib.pop(qn(attr), None)


def _para(style, align=None, before=0, after=0, indent=None, keep_next=False, line=1.0):
    pf = style.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), line
    if align is not None:
        pf.alignment = align
    if indent is not None:
        pf.first_line_indent = indent
    pf.keep_with_next = keep_next


def make_reference(path):
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(path))
    by_name = {sty.name.lower(): sty for sty in doc.styles}

    class _St:
        def __getitem__(self, key):
            return by_name[key.lower()]

    st = _St()
    for name in ("Normal", "Body Text", "First Paragraph", "Compact", "Block Text", "Bibliography"):
        _font(st[name], 10)
        _para(st[name], WD_ALIGN_PARAGRAPH.JUSTIFY, indent=Cm(0.35) if name == "Body Text" else Cm(0))
    _para(st["Compact"], WD_ALIGN_PARAGRAPH.JUSTIFY, after=1)
    _font(st["Title"], 22)
    _para(st["Title"], WD_ALIGN_PARAGRAPH.CENTER, before=0, after=4)
    _font(st["Subtitle"], 11, italic=True)
    _para(st["Subtitle"], WD_ALIGN_PARAGRAPH.CENTER, after=8)
    _font(st["Heading 1"], 10, small_caps=True)
    _para(st["Heading 1"], WD_ALIGN_PARAGRAPH.CENTER, before=10, after=5, keep_next=True)
    _font(st["Heading 2"], 10, italic=True)
    _para(st["Heading 2"], WD_ALIGN_PARAGRAPH.LEFT, before=8, after=3, keep_next=True)
    _font(st["Heading 3"], 10, italic=True)
    _para(st["Heading 3"], WD_ALIGN_PARAGRAPH.LEFT, before=6, after=2, keep_next=True)
    for name in ("Caption", "Image Caption"):
        _font(st[name], 8)
        _para(st[name], WD_ALIGN_PARAGRAPH.JUSTIFY, before=2, after=8)
    _font(st["Table Caption"], 8, small_caps=True)
    _para(st["Table Caption"], WD_ALIGN_PARAGRAPH.CENTER, before=6, after=3, keep_next=True)
    for name in ("Figure", "Captioned Figure"):
        _para(st[name], WD_ALIGN_PARAGRAPH.CENTER, before=6, after=0, keep_next=True)
    if "hyperlink" in by_name:
        by_name["hyperlink"].font.color.rgb = BLACK
        by_name["hyperlink"].font.underline = False
    if "verbatim char" in by_name:
        by_name["verbatim char"].font.size = Pt(8.5)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = PAGE_W, PAGE_H
    sec.top_margin = sec.bottom_margin = MARGIN_TB
    sec.left_margin = sec.right_margin = MARGIN_LR
    doc.save(str(path))


def _sect(cols, continuous=True):
    """A w:sectPr for A4 with the given number of columns."""
    s = OxmlElement("w:sectPr")
    t = OxmlElement("w:type")
    t.set(qn("w:val"), "continuous" if continuous else "nextPage")
    s.append(t)
    sz = OxmlElement("w:pgSz")
    sz.set(qn("w:w"), str(int(PAGE_W.twips)))
    sz.set(qn("w:h"), str(int(PAGE_H.twips)))
    s.append(sz)
    mar = OxmlElement("w:pgMar")
    for k, v in (("top", MARGIN_TB), ("bottom", MARGIN_TB), ("left", MARGIN_LR), ("right", MARGIN_LR),
                 ("header", Cm(1.0)), ("footer", Cm(1.0)), ("gutter", 0)):
        mar.set(qn(f"w:{k}"), str(int(v.twips) if v else 0))
    s.append(mar)
    c = OxmlElement("w:cols")
    c.set(qn("w:num"), str(cols))
    c.set(qn("w:space"), str(int(GAP.twips)))
    s.append(c)
    return s


def _end_section_at(paragraph_el, cols):
    """Make paragraph_el the last paragraph of a section with `cols` columns."""
    ppr = paragraph_el.find(qn("w:pPr"))
    if ppr is None:
        ppr = OxmlElement("w:pPr")
        paragraph_el.insert(0, ppr)
    old = ppr.find(qn("w:sectPr"))
    if old is not None:
        ppr.remove(old)
    ppr.append(_sect(cols))


def _empty_para_after(el):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "0")
    sp.set(qn("w:after"), "0")
    sp.set(qn("w:line"), "20")
    sp.set(qn("w:lineRule"), "exact")
    ppr.append(sp)
    p.append(ppr)
    el.addnext(p)
    return p


def _empty_para_before(el):
    p = _empty_para_after(el)
    el.addprevious(p)
    return p


def _border(parent, edge, size):
    b = OxmlElement(f"w:{edge}")
    b.set(qn("w:val"), "single" if size else "nil")
    if size:
        b.set(qn("w:sz"), str(size))
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "000000")
    parent.append(b)


def _three_rule(table):
    """Top rule, thin rule under the header, bottom rule; nothing else."""
    tbl = table._tbl
    tblpr = tbl.tblPr
    for tag in ("w:tblStyle", "w:tblBorders", "w:tblLook"):
        el = tblpr.find(qn(tag))
        if el is not None:
            tblpr.remove(el)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        _border(borders, edge, 0)
    tblpr.append(borders)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    rows = table.rows
    for r_i, row in enumerate(rows):
        for cell in row.cells:
            tcpr = cell._tc.get_or_add_tcPr()
            old = tcpr.find(qn("w:tcBorders"))
            if old is not None:
                tcpr.remove(old)
            shd = tcpr.find(qn("w:shd"))
            if shd is not None:
                tcpr.remove(shd)
            cb = OxmlElement("w:tcBorders")
            if r_i == 0:
                _border(cb, "top", 12)
            if r_i == 0 and len(rows) > 1:
                _border(cb, "bottom", 4)
            if r_i == len(rows) - 1:
                _border(cb, "bottom", 12)
            tcpr.append(cb)
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(1)
                for run in p.runs:
                    run.font.size = Pt(8)
                    if r_i == 0:
                        run.font.bold = False


def _page_numbers(doc):
    for sec in doc.sections:
        sec.footer.is_linked_to_previous = True
    footer = doc.sections[0].footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), kind)
            run._r.append(fc)
        else:
            it = OxmlElement("w:instrText")
            it.set(qn("xml:space"), "preserve")
            it.text = text
            run._r.append(it)
    run.font.size = Pt(9)


def finish(docx_path, wide_images):
    """wide_images: one bool per image in document order (True = spans both columns)."""
    doc = Document(str(docx_path))
    body = doc.element.body

    # images: column width, or full text width for wide ones; remember the wide figure paragraphs
    wide_blocks = []
    extents = body.findall(".//" + qn("wp:inline"))
    for i, inline in enumerate(extents):
        wide = wide_images[i] if i < len(wide_images) else False
        ext = inline.find(qn("wp:extent"))
        cx, cy = int(ext.get("cx")), int(ext.get("cy"))
        target = int(TEXT_W * 0.92) if wide else int(COL_W)
        if wide and i < len(wide_images) and isinstance(wide_images[i], float):
            target = int(TEXT_W * wide_images[i])
        scale = target / cx
        new_cx, new_cy = int(cx * scale), int(cy * scale)
        ext.set("cx", str(new_cx))
        ext.set("cy", str(new_cy))
        for a_ext in inline.findall(".//" + qn("a:ext")):
            a_ext.set("cx", str(new_cx))
            a_ext.set("cy", str(new_cy))
        if wide:
            para = inline
            while para.tag != qn("w:p"):
                para = para.getparent()
            wide_blocks.append(para)

    # paired images in one paragraph (confusion matrices): half text width each, whole paragraph wide
    for p in body.iter(qn("w:p")):
        inl = p.findall(".//" + qn("wp:inline"))
        if len(inl) == 2:
            for inline in inl:
                ext = inline.find(qn("wp:extent"))
                cx, cy = int(ext.get("cx")), int(ext.get("cy"))
                tgt = int(TEXT_W * 0.45)
                for e in [ext] + inline.findall(".//" + qn("a:ext")):
                    e.set("cx", str(tgt))
                    e.set("cy", str(int(cy * tgt / cx)))
            if p not in wide_blocks:
                wide_blocks.append(p)

    for table in doc.tables:
        _three_rule(table)

    # placeholders: grey highlight instead of yellow (black-and-white layout)
    for hl in body.iter(qn("w:highlight")):
        hl.set(qn("w:val"), "lightGray")

    # title block: everything before the first Heading 1 is centred and single-column
    first_h1 = None
    for p in doc.paragraphs:
        if p.style.name.lower() == "heading 1":
            first_h1 = p
            break
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0)
        if p.style.name.lower() not in ("title", "subtitle"):
            for run in p.runs:
                run.font.size = Pt(9.5)
    if first_h1 is not None:
        _end_section_at(_empty_para_before(first_h1._p), 1)

    # wide figures: [image paragraph + caption] in a single-column section
    for para in wide_blocks:
        nxt = para.getnext()
        last = nxt if nxt is not None and nxt.tag == qn("w:p") and _style(nxt) in ("ImageCaption", "Caption") else para
        _end_section_at(_empty_para_before(para), 2)
        _end_section_at(_empty_para_after(last), 1)

    # wide tables (4+ columns): [caption + table] in a single-column section
    for table in doc.tables:
        if len(table.columns) < 4:
            continue
        tbl = table._tbl
        prev = tbl.getprevious()
        start = prev if prev is not None and prev.tag == qn("w:p") and _style(prev) == "TableCaption" else tbl
        _end_section_at(_empty_para_before(start), 2)
        _end_section_at(_empty_para_after(tbl), 1)

    # body ends in two columns
    final = body.find(qn("w:sectPr"))
    if final is not None:
        body.remove(final)
    body.append(_sect(2))
    _page_numbers(doc)
    doc.save(str(docx_path))


def _style(p_el):
    ppr = p_el.find(qn("w:pPr"))
    if ppr is None:
        return ""
    ps = ppr.find(qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else ""
