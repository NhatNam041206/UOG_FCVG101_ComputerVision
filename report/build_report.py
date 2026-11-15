"""Build the coursework report (PDF + editable .docx) from report.md.

Run from the project root:   python report/build_report.py              (.docx only, without source code)
                             python report/build_report.py --pdf        (also the PDF)
                             python report/build_report.py --with-code  (adds Appendix A: source code)
Needs: pandoc, matplotlib, and Node Playwright with Chromium (HTML -> PDF).
Outputs: report/FCVG101_Coursework_Report[_with_code_appendix].pdf and .docx

Conventions in report.md:
  $$ ... $$ {#3.1}   numbered display equation (rendered to SVG for the PDF, editable maths in Word)
  ::: {.algorithm}   framed algorithm box
  : Table 3.1: ...   table caption;  ![Figure 3.1: ...](path)  figure with caption
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
BUILD = REPORT / "build"
NAME = "FCVG101_Coursework_Report"
TITLE = "A Vision Inspection Assistant for an Organic Flower Farm"
SUBTITLE = "FCVG101 Introduction to Computer Vision: Coursework Report"

CODE_SECTIONS = [
    ("Configuration and entry point", ["requirements.txt", "configs/config.yaml", "app.py"]),
    ("Shared helpers", ["src/__init__.py", "src/config.py", "src/io_utils.py", "src/visualize.py"]),
    ("Task 1A: calibration", ["src/calibration.py"]),
    ("Task 1B: classical processing", ["src/filtering.py", "src/frequency.py", "src/edges.py", "src/features.py",
                                       "src/task1b.py"]),
    ("Task 2: data pipeline, model, training, evaluation", ["src/data_pipeline.py", "src/model.py", "src/train.py",
                                                             "src/evaluate.py", "src/baselines.py", "src/explain.py"]),
    ("Task 3: assistant", ["src/inference.py", "src/assistant.py"]),
    ("Unit tests", sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tests").glob("test_*.py"))),
    ("Supporting capture tools (Task 1A)", ["tools/task1a/capture_checker_task1a.py",
                                            "tools/task1a/undistort_view_task1a.py"]),
]
LANG = {".py": "python", ".yaml": "yaml", ".txt": "text"}

CSS = """
/* Conference-paper layout (IEEE-like): A4, two columns, Times 10 pt, black and white, three-rule tables. */
@page { size: A4; margin: 19mm 15mm 22mm 15mm; }
body { font-family: 'Liberation Serif', 'Times New Roman', 'DejaVu Serif', serif; font-size: 10pt; line-height: 1.2;
       color: #000; max-width: none; margin: 0; padding: 0; text-align: justify; hyphens: auto;
       column-count: 2; column-gap: 6mm; column-fill: auto; }
p { margin: 0; text-indent: 1em; }
h1 + p, h2 + p, h3 + p, figure + p, table + p, .eqn + p, .algorithm + p, ul + p, ol + p, .figpair + p { text-indent: 0; }
.cover { column-span: all; text-align: center; margin: 0 0 6mm; }
.cover p { text-indent: 0; text-align: center; }
.cover h1.title { font-size: 22pt; font-weight: normal; font-variant: normal; letter-spacing: 0; margin: 2mm 0 2mm; line-height: 1.15; }
.cover .subtitle { font-size: 11pt; font-style: italic; margin: 0 auto 4mm; max-width: 150mm; }
.cover .author { font-size: 10pt; line-height: 1.3; margin-bottom: 3mm; }
.cover .links { font-size: 9pt; line-height: 1.3; }
.cover .note { font-size: 8pt; font-style: italic; margin-top: 2mm; }
h1 { font-size: 10pt; font-weight: normal; font-variant: small-caps; letter-spacing: 0.4px;
     text-align: center; margin: 10pt 0 5pt; break-after: avoid; page-break-after: avoid; }
h2 { font-size: 10pt; font-weight: normal; font-style: italic; text-align: left; margin: 8pt 0 3pt; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 10pt; font-style: italic; font-weight: normal; margin: 6pt 0 2pt; break-after: avoid; }
ul, ol { margin: 3pt 0; padding-left: 1.4em; }
li { margin: 0 0 1pt; }
a { color: #000; text-decoration: none; }
mark { background: #e3e3e3; padding: 0 1px; }
code { font-family: 'DejaVu Sans Mono', 'Courier New', monospace; font-size: 8.3pt; }
table { border-collapse: collapse; margin: 4pt auto 8pt; font-size: 8pt; line-height: 1.15; width: 100%;
        break-inside: avoid; page-break-inside: avoid; text-align: left; }
table.wide { column-span: all; }
th, td { border: none; padding: 1.5pt 4pt; vertical-align: top; background: none; }
thead tr { border-top: 1.2pt solid #000; border-bottom: 0.6pt solid #000; }
thead th { font-weight: normal; }
tbody tr:last-child { border-bottom: 1.2pt solid #000; }
caption { caption-side: top; font-size: 8pt; font-variant: small-caps; text-align: center; padding-bottom: 3pt; color: #000; font-weight: normal; }
figure { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; page-break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figure:not(.wide) img { width: 100% !important; }
figure.wide { column-span: all; }
figcaption, .caption { font-size: 8pt; text-align: justify; margin-top: 3pt; text-indent: 0; color: #000; }
.figpair { column-span: all; text-align: center; break-inside: avoid; margin: 6pt 0 8pt; }
.figpair p { text-indent: 0; }
.eqn { display: flex; align-items: center; margin: 5pt 0; break-inside: avoid; }
.eqn .eqbody { flex: 1; text-align: center; }
.eqn .eqbody img { max-width: 100%; vertical-align: middle; }
.eqn .eqno { width: 30px; text-align: right; }
.algorithm { border-top: 1.2pt solid #000; border-bottom: 1.2pt solid #000; margin: 6pt 0; padding: 1pt 0 2pt; break-inside: avoid; font-size: 9pt; }
.algorithm p { text-indent: 0; }
.algorithm p:first-child { border-bottom: 0.6pt solid #000; padding-bottom: 2pt; margin-bottom: 2pt; }
.algorithm ol { margin: 2pt 0; }
.repo-link { text-align: center; text-indent: 0; margin: 4pt 0; font-size: 9pt; word-break: break-all; }
h1#references + ol { font-size: 8.5pt; padding-left: 1.6em; }
pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 6.8pt; line-height: 1.2; text-align: left; white-space: pre-wrap;
      word-break: break-all; border-top: 0.6pt solid #000; border-bottom: 0.6pt solid #000; padding: 3pt 0; }
pre code { font-size: 6.8pt; }
pre > code.sourceCode > span { display: inline !important; text-indent: 0 !important; padding-left: 0 !important; }
h3.codefile { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; margin: 8pt 0 0; }
"""

EQ = re.compile(r"\$\$(.+?)\$\$\s*\{#([\d.]+)\}", flags=re.S)


def mathtext(tex):
    """Translate the LaTeX used in report.md into matplotlib mathtext."""
    tex = re.sub(r"\\text\{([^}]*)\}", lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", tex)
    tex = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", tex)
    for old, new in ((r"\lVert", r"\Vert"), (r"\rVert", r"\Vert"), (r"\Big", ""), (r"\big", ""),
                     (r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\tfrac", r"\frac"), (r"\frac", r"\dfrac")):
        tex = tex.replace(old, new)
    for short, full in (("le", "leq"), ("ge", "geq"), ("ne", "neq")):
        tex = re.sub(rf"\\{short}(?![a-zA-Z])", lambda m: "\\" + full, tex)
    return tex


def render_equations(markdown):
    """PDF: each numbered $$...$$ becomes an SVG picture with its number on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    matplotlib.rcParams["svg.fonttype"] = "path"

    def draw(match):
        number = match.group(2)
        name = f"eq_{number.replace('.', '_')}.svg"
        fig = plt.figure()
        fig.text(0, 0, "$" + mathtext(match.group(1).strip()) + "$", fontsize=10)
        fig.savefig(BUILD / name, bbox_inches="tight", pad_inches=0.03, transparent=True)
        plt.close(fig)
        return (f'<div class="eqn"><span class="eqbody"><img src="{name}" alt="Equation {number}"/></span>'
                f'<span class="eqno">({number})</span></div>')

    return EQ.sub(draw, markdown)


def word_equations(markdown):
    """Word: keep editable maths and append the equation number."""
    return EQ.sub(lambda m: f"$${m.group(1).strip()} \\qquad ({m.group(2)})$$", markdown)


def word_cover(cover):
    """Word drops raw HTML, so turn the cover's HTML lines into plain Markdown."""
    cover = re.sub(r'<h1 class="title">.*?</h1>', "", cover)
    cover = cover.replace("<br/>", "  \n")
    cover = re.sub(r'<a href="[^"]*">(.*?)</a>', r"\1", cover)
    cover = re.sub(r'<p class="[^"]*">(.*?)</p>', r"\1", cover, flags=re.S)
    return re.sub(r"</?div[^>]*>", "", cover)


def widen(html_path):
    """Two-column layout: figures set wider than 70 % and tables with 4+ columns span both columns."""
    html = html_path.read_text(encoding="utf-8")

    def fig(m):
        widths = [float(w) for w in re.findall(r"width:\s*([\d.]+)%", m.group(0))]
        return m.group(0).replace("<figure>", '<figure class="wide">', 1) if widths and max(widths) > 70 else m.group(0)

    html = re.sub(r"<figure>.*?</figure>", fig, html, flags=re.S)

    def tab(m):
        head = re.search(r"<tr[^>]*>(.*?)</tr>", m.group(0), flags=re.S)
        cols = len(re.findall(r"<t[hd][ >]", head.group(1))) if head else 0
        return m.group(0).replace("<table>", '<table class="wide">', 1) if cols >= 4 else m.group(0)

    html = re.sub(r"<table>.*?</table>", tab, html, flags=re.S)
    html_path.write_text(html, encoding="utf-8")


def code_appendix():
    lines = ["", "# Appendix A: Source code", "",
             "All Python source and configuration files of the project, reproduced as selectable text. The notebook "
             "only calls these functions and is not repeated. The course's original workshop script "
             "`tools/task1a/calibration_task1a.py` is not reproduced.", ""]
    for number, (title, files) in enumerate(CODE_SECTIONS, start=1):
        lines += [f"## A.{number} {title}", ""]
        for rel in files:
            path = ROOT / rel
            if not path.is_file():
                lines += [f"*{rel}: file not found when the report was built.*", ""]
                continue
            text = path.read_text(encoding="utf-8-sig").rstrip("\n")
            fence = "````" if "```" in text else "```"
            lines += [f'### `{rel}` ({text.count(chr(10)) + 1} lines) {{.codefile}}', "",
                      f"{fence}{LANG.get(path.suffix, 'text')}", text, fence, ""]
    return "\n".join(lines)


def build_pdf(name, cover, body, appendix):
    # image paths in report.md are relative to report/; the HTML is built inside report/build/
    html_body = re.sub(r"(\]\(|src=\")(diagrams|figures)/", r"\1../\2/", render_equations(body)) + "\n" + appendix
    (BUILD / "report_html.md").write_text(html_body, encoding="utf-8")
    (BUILD / "cover.md").write_text(cover, encoding="utf-8")
    subprocess.run(["pandoc", "cover.md", "-f", "markdown", "-o", "cover.html"], check=True, cwd=BUILD)
    (BUILD / "style.css").write_text(CSS, encoding="utf-8")
    subprocess.run(["pandoc", "report_html.md", "-o", "report.html", "--standalone", "--mathml",
                    "--highlight-style=tango", "--css", "style.css", "--embed-resources",
                    "--metadata", "lang=en-GB", "--metadata", f"pagetitle={TITLE}",
                    "--include-before-body", "cover.html"], check=True, cwd=BUILD)
    widen(BUILD / "report.html")
    pdf = REPORT / f"{name}.pdf"
    footer = ('<div style="font-size:7.5pt;width:100%;text-align:center;color:#000;font-family:Liberation Serif">'
              '<span class="pageNumber"></span></div>')
    script = BUILD / "to_pdf.js"
    script.write_text(f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto({json.dumps((BUILD / 'report.html').as_uri())}, {{ waitUntil: 'networkidle' }});
  await page.pdf({{ path: {json.dumps(str(pdf))}, format: 'A4', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>', footerTemplate: {json.dumps(footer)},
    margin: {{ top: '19mm', bottom: '22mm', left: '15mm', right: '15mm' }} }});
  await browser.close();
}})();
""", encoding="utf-8")
    subprocess.run(["node", str(script)], check=True)
    print(f"{pdf}: {pdf.stat().st_size / 1e6:.2f} MB")


def word_body(md):
    """Word drops raw HTML: turn the few HTML snippets in the body into Markdown."""
    md = re.sub(r'<a href="([^"]*)">(.*?)</a>', r"[\2](\1)", md)
    md = re.sub(r'<p class="[^"]*">(.*?)</p>', r"\1", md, flags=re.S)
    md = re.sub(r"</?div[^>]*>", "", md)
    md = re.sub(r"<mark>(.*?)</mark>", r"[\1]{.mark}", md)
    return md


def build_docx(name, cover, body, appendix):
    import docx_style
    ref = BUILD / "reference.docx"
    docx_style.make_reference(ref)
    subtitle = re.search(r'<p class="subtitle">(.*?)</p>', cover).group(1)
    cover = re.sub(r'<p class="subtitle">.*?</p>', "", cover)
    md = word_body(word_cover(cover)) + "\n" + word_body(word_equations(body)).replace(".svg)", ".png)") + "\n" + appendix
    (BUILD / "report_docx.md").write_text(md, encoding="utf-8")
    widths = [float(w) if w else 0.0 for w in re.findall(r"!\[[^\]]*\]\([^)]*\)(?:\{width=([\d.]+)%\})?", md)]
    wide = [w > 70 for w in widths]
    out = REPORT / f"{name}.docx"
    subprocess.run(["pandoc", str(BUILD / "report_docx.md"), "-o", str(out), "--reference-doc", str(ref),
                    "--metadata", f"title={TITLE}", "--metadata", f"subtitle={subtitle}",
                    "--highlight-style=tango", "--resource-path", str(REPORT)], check=True)
    docx_style.finish(out, wide)
    print(f"{out}: {out.stat().st_size / 1e6:.2f} MB")


def main():
    """Default: .docx only (nam converts it to PDF). --pdf also builds the PDF; --with-code adds the source appendix."""
    with_code = "--with-code" in sys.argv
    name = NAME + ("_with_code_appendix" if with_code else "")
    BUILD.mkdir(exist_ok=True)
    source = (REPORT / "report.md").read_text(encoding="utf-8")
    cover, body = source.split("<!-- END-COVER -->", 1)
    appendix = code_appendix() if with_code else ""
    build_docx(name, cover, body, appendix)
    if "--pdf" in sys.argv:
        build_pdf(name, cover, body, appendix)


if __name__ == "__main__":
    main()
