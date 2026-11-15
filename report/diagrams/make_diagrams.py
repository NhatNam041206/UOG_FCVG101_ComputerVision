"""Draw the report's UML diagrams as SVG (no external tools), then build/diagrams.html for PNG export.

Run: python report/diagrams/make_diagrams.py   (writes report/diagrams/*.svg)
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "DejaVu Sans, Arial, sans-serif"
INK, FILL, ACC, SOFT = "#000000", "#ffffff", "#000000", "#efefef"


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, label, size=13, anchor="middle", weight="normal", color=INK, italic=False):
    lines = label.split("\n")
    top = y - (len(lines) - 1) * size * 0.62
    style = ' font-style="italic"' if italic else ""
    return "".join(f'<text x="{x}" y="{top + i * size * 1.24:.1f}" font-size="{size}" text-anchor="{anchor}" '
                   f'dominant-baseline="middle" font-weight="{weight}" fill="{color}"{style}>{esc(l)}</text>'
                   for i, l in enumerate(lines))


def box(x, y, w, h, label, fill=FILL, rx=8, size=13, weight="normal", stroke=INK, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.6"{d}/>' + text(x + w / 2, y + h / 2, label, size, weight=weight))


def component(x, y, w, h, title, sub=""):
    icon = (f'<rect x="{x + w - 30}" y="{y + 8}" width="20" height="14" fill="white" stroke="{INK}"/>'
            f'<rect x="{x + w - 34}" y="{y + 11}" width="8" height="3" fill="white" stroke="{INK}"/>'
            f'<rect x="{x + w - 34}" y="{y + 16}" width="8" height="3" fill="white" stroke="{INK}"/>')
    body = text(x + w / 2, y + h / 2 - (8 if sub else 0), title, 13.5, weight="bold")
    if sub:
        body += text(x + w / 2, y + h / 2 + 16 + sub.count("\n") * 8, sub, 11.5, color="#333333")
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{FILL}" stroke="{INK}" stroke-width="1.6"/>'
            + icon + body)


def artifact(x, y, w, h, label):
    fold = 14
    return (f'<path d="M{x},{y} H{x + w - fold} L{x + w},{y + fold} V{y + h} H{x} Z" fill="{SOFT}" stroke="{INK}" '
            f'stroke-width="1.5"/><path d="M{x + w - fold},{y} V{y + fold} H{x + w}" fill="none" stroke="{INK}" '
            f'stroke-width="1.5"/>' + text(x + w / 2, y + 13, "«artifact»", 10, color="#444444")
            + text(x + w / 2, y + h / 2 + 7, label, 12))


def arrow(x1, y1, x2, y2, label="", dash=False, color=INK, lx=None, ly=None, size=11.5, anchor="middle", head=True,
          marker="arr"):
    d = ' stroke-dasharray="6 4"' if dash else ""
    m = f' marker-end="url(#{marker})"' if head else ""
    out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"{d}{m}/>'
    if label:
        out += text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 9, label,
                    size, anchor=anchor, color="#222222")
    return out


def poly(points, label="", dash=False, lx=0, ly=0, anchor="start", marker="arr"):
    d = ' stroke-dasharray="6 4"' if dash else ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    out = f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.5"{d} marker-end="url(#{marker})"/>'
    if label:
        out += text(lx, ly, label, 11.5, anchor=anchor, color="#222222")
    return out


def note(x, y, w, h, label, size=11.5):
    fold = 14
    return (f'<path d="M{x},{y} H{x + w - fold} L{x + w},{y + fold} V{y + h} H{x} Z" fill="{SOFT}" stroke="{INK}" '
            f'stroke-width="1.4"/><path d="M{x + w - fold},{y} V{y + fold} H{x + w}" fill="none" stroke="{INK}" '
            f'stroke-width="1.4"/>' + text(x + w / 2, y + h / 2, label, size))


def merge(x, y, s=16):
    return (f'<polygon points="{x},{y - s} {x + s},{y} {x},{y + s} {x - s},{y}" fill="{SOFT}" stroke="{INK}" '
            f'stroke-width="1.6"/>')


def start(x, y):
    return f'<circle cx="{x}" cy="{y}" r="10" fill="{INK}"/>'


def end(x, y):
    return f'<circle cx="{x}" cy="{y}" r="12" fill="white" stroke="{INK}" stroke-width="1.6"/><circle cx="{x}" cy="{y}" r="7" fill="{INK}"/>'


def diamond(x, y, label, w=150, h=56):
    return (f'<polygon points="{x},{y - h / 2} {x + w / 2},{y} {x},{y + h / 2} {x - w / 2},{y}" fill="{SOFT}" '
            f'stroke="{INK}" stroke-width="1.6"/>' + text(x, y, label, 12))


def actor(x, y, label):
    return (f'<circle cx="{x}" cy="{y}" r="13" fill="white" stroke="{INK}" stroke-width="1.8"/>'
            f'<line x1="{x}" y1="{y + 13}" x2="{x}" y2="{y + 50}" stroke="{INK}" stroke-width="1.8"/>'
            f'<line x1="{x - 22}" y1="{y + 26}" x2="{x + 22}" y2="{y + 26}" stroke="{INK}" stroke-width="1.8"/>'
            f'<line x1="{x}" y1="{y + 50}" x2="{x - 18}" y2="{y + 78}" stroke="{INK}" stroke-width="1.8"/>'
            f'<line x1="{x}" y1="{y + 50}" x2="{x + 18}" y2="{y + 78}" stroke="{INK}" stroke-width="1.8"/>'
            + text(x, y + 98, label, 13, weight="bold"))


def svg(name, width, height, body, title):
    top = 30
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height - top}" '
           f'viewBox="0 {top} {width} {height - top}" '
           f'font-family="{FONT}"><defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" '
           f'markerHeight="9" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
           f'<marker id="open" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
           f'orient="auto"><path d="M0,0 L10,5 L0,10" fill="none" stroke="{INK}"/></marker></defs>'
           f'<rect x="0" y="{top}" width="{width}" height="{height}" fill="white"/>' + body + "</svg>")
    (OUT / f"{name}.svg").write_text(doc, encoding="utf-8")


# ---------------------------------------------------------------- 1. Use case diagram
def use_case():
    b = actor(95, 230, "Farm manager")
    b += actor(1005, 120, "Webcam\n(secondary actor)")
    b += f'<rect x="230" y="50" width="640" height="560" rx="14" fill="white" stroke="{ACC}" stroke-width="2"/>'
    b += text(550, 74, "Vision Inspection Assistant (system boundary)", 13.5, weight="bold", color=ACC)
    cases = [("Calibrate the camera", "Task 1A", 125), ("Correct lens distortion\nof a photo", "Task 1A", 200),
             ("Remove noise from a photo", "Task 1B", 275), ("Analyse image frequencies", "Task 1B", 350),
             ("Find edges", "Task 1B", 425), ("Match two photos\nof the same scene", "Task 1B", 500),
             ("Identify the flower type", "Task 2", 575)]
    for label, task, y in cases:
        b += f'<ellipse cx="500" cy="{y}" rx="165" ry="30" fill="{FILL}" stroke="{INK}" stroke-width="1.6"/>'
        b += text(500, y, label, 12.5)
        b += text(760, y, task, 11.5, color="#555555", italic=True)
        b += f'<line x1="120" y1="265" x2="337" y2="{y}" stroke="{INK}" stroke-width="1.3"/>'
    b += f'<line x1="980" y1="150" x2="665" y2="125" stroke="{INK}" stroke-width="1.3"/>'
    b += poly([(665, 200), (720, 200), (720, 140), (667, 132)], dash=True, marker="open")
    b += text(770, 168, "«include»", 11.5, color="#222222")
    svg("uml_use_case", 1080, 630, b, "UML use case diagram: what a user can ask the assistant to do")


# ---------------------------------------------------------------- 2. Component diagram
def components():
    # UML component diagram. Dashed open arrows are dependencies («use», «create»); artifacts carry «artifact».
    b = component(330, 50, 340, 70, "Web app (user interface)", "upload photo · choose a task · see results")
    b += component(330, 175, 340, 70, "Assistant (request router)", "validates the request, dispatches to an engine")
    b += component(40, 320, 280, 80, "Camera calibration", "Task 1A")
    b += component(360, 320, 280, 80, "Classical image analysis", "Task 1B: filters, frequency,\nedges, features")
    b += component(680, 320, 280, 80, "Flower recognition", "Task 2: inference with the\ntrained model")
    b += artifact(70, 455, 220, 62, "Camera parameters\n(K + distortion)")
    b += artifact(710, 455, 220, 62, "Trained model weights\n+ class list")
    b += component(680, 555, 280, 85, "Training pipeline", "Task 2: clean, split, train,\nevaluate, select")
    b += artifact(390, 455, 220, 62, "Configuration file\n(all parameters)")
    b += artifact(70, 570, 220, 56, "Checkerboard images")
    b += artifact(390, 570, 220, 56, "Flower image dataset")
    b += arrow(500, 120, 500, 173, "«use»", lx=512, anchor="start", dash=True, marker="open")
    b += arrow(420, 245, 180, 318, "«use»", lx=262, ly=272, dash=True, marker="open")
    b += arrow(500, 245, 500, 318, "«use»", lx=512, ly=282, anchor="start", dash=True, marker="open")
    b += arrow(580, 245, 820, 318, "«use»", lx=742, ly=272, dash=True, marker="open")
    b += arrow(180, 400, 180, 453, "«create»", lx=190, anchor="start", dash=True, marker="open")
    b += arrow(820, 400, 820, 453, "«use»", lx=830, ly=428, anchor="start", dash=True, marker="open")
    b += arrow(820, 555, 820, 519, "«create»", lx=830, ly=540, anchor="start", dash=True, marker="open")
    b += poly([(40, 360), (25, 360), (25, 598), (68, 598)], "«use»", dash=True, marker="open", lx=30, ly=480)
    b += poly([(680, 598), (612, 598)], dash=True, marker="open")
    b += text(644, 584, "«use»", 11, color="#222222")
    b += arrow(500, 400, 500, 453, "", dash=True, marker="open")
    b += text(512, 430, "«use» (by every component)", 11, anchor="start", color="#222222")
    svg("uml_components", 1000, 650, b, "")


# ---------------------------------------------------------------- 3. Activity: calibration
def activity_calibration():
    # UML activity diagram with a per-image loop (decision + merge) and a guard on the minimum number of views.
    cx, w = 330, 400
    b = start(cx, 50) + arrow(cx, 60, cx, 78)
    b += box(cx - w / 2, 80, w, 46, "Capture 13 checkerboard images (guided capture tool)")
    b += arrow(cx, 126, cx, 148)
    b += merge(cx, 164)
    b += arrow(cx, 180, cx, 198)
    b += box(cx - w / 2, 200, w, 46, "Detect the 6 × 9 inner corners in the next image")
    b += arrow(cx, 246, cx, 266)
    b += diamond(cx, 296, "corners\nfound?", w=140, h=56)
    b += arrow(cx + 70, 296, 588, 296, "[no]", lx=540)
    b += box(590, 274, 160, 44, "Discard the image", fill=SOFT)
    b += arrow(cx, 324, cx, 346, "[yes]", lx=cx + 10, ly=334, anchor="start")
    b += box(cx - w / 2, 348, w, 46, "Refine corners to sub-pixel accuracy and store them")
    b += arrow(cx, 394, cx, 414)
    b += merge(cx, 430)
    b += poly([(670, 318), (670, 430), (cx + 18, 430)])
    b += arrow(cx, 446, cx, 466)
    b += diamond(cx, 496, "more\nimages?", w=140, h=56)
    b += poly([(cx - 70, 496), (60, 496), (60, 164), (cx - 18, 164)], "[yes]", lx=70, ly=484)
    b += arrow(cx, 524, cx, 546, "[no]", lx=cx + 10, ly=534, anchor="start")
    b += diamond(cx, 576, "≥ 3 valid\nviews?", w=140, h=56)
    b += arrow(cx + 70, 576, 588, 576, "[no]", lx=540)
    b += box(590, 554, 160, 44, "Report an error", fill=SOFT)
    b += arrow(cx, 604, cx, 626, "[yes]", lx=cx + 10, ly=614, anchor="start")
    rest = ["Initialise K from the board-to-image homographies (closed form)",
            "Refine K, distortion and poses jointly (Levenberg–Marquardt)",
            "Compute the RMS re-projection error, overall and per view",
            "Undistort a sample image and save the camera parameters"]
    y = 628
    for i, label in enumerate(rest):
        b += box(cx - w / 2 - 40, y, w + 80, 44, label)
        b += arrow(cx, y + 44, cx, y + 62)
        y += 64
    b += end(cx, y + 10)
    b += poly([(670, 598), (670, y + 10), (cx + 14, y + 10)])
    svg("uml_activity_calibration", 800, y + 36, b, "")


# ---------------------------------------------------------------- 4. Activity: classical analysis
def activity_classical():
    W = 1060
    b = start(W / 2, 48) + arrow(W / 2, 58, W / 2, 78)
    b += box(W / 2 - 290, 80, 580, 46, "Load the test photos (one scene + a pair + an unrelated photo)")
    b += arrow(W / 2, 126, W / 2, 146)
    b += f'<rect x="60" y="148" width="{W - 120}" height="7" fill="{INK}"/>'
    lanes = [("Noise removal", ["Add known noise\n(grainy / speckled)", "Apply box, Gaussian\nand median filters",
                                "Score each against\nthe clean photo (PSNR)"]),
             ("Frequency analysis", ["Convert photo to\nfrequencies (FFT)", "Keep low or high\nfrequencies (masks)",
                                     "Convert back and\ncompare results"]),
             ("Edges", ["Smooth slightly", "Measure brightness\nchange (Sobel)", "Keep thin, connected\nedges (Canny)"]),
             ("Feature matching", ["Find distinctive points\n(Harris / SIFT / ORB)", "Pair points with the\nratio test",
                                   "Keep pairs that agree\non one geometry (RANSAC)"])]
    lw = 225
    for i, (title, items) in enumerate(lanes):
        x = 40 + i * (lw + 25) + 12
        b += arrow(x + lw / 2, 155, x + lw / 2, 178)
        b += text(x + lw / 2, 192, title, 13, weight="bold", color=ACC)
        y = 208
        for j, item in enumerate(items):
            b += box(x, y, lw, 54, item, size=12)
            if j < len(items) - 1:
                b += arrow(x + lw / 2, y + 54, x + lw / 2, y + 74)
            y += 76
        b += arrow(x + lw / 2, y - 22, x + lw / 2, 440)
    b += f'<rect x="60" y="442" width="{W - 120}" height="7" fill="{INK}"/>'
    b += arrow(W / 2, 449, W / 2, 470)
    b += box(W / 2 - 200, 472, 400, 46, "Save every picture and score for the report")
    b += arrow(W / 2, 518, W / 2, 538) + end(W / 2, 550)
    svg("uml_activity_classical", W, 575, b, "UML activity diagram: classical image analysis (Task 1B), four independent branches")


# ---------------------------------------------------------------- 5. Activity: data pipeline + training
def activity_training():
    # UML activity diagram: linear pipeline, then a loop over candidate models expressed with merge + decision nodes.
    cx, w = 360, 470
    b = start(cx, 42)
    rows = [("1  Ingest: download 3,670 flower photos in 5 class folders", 70),
            ("2  Inventory: record label, size, colour mode and readability", 132),
            ("3  Clean: remove corrupt, tiny, extreme-aspect and duplicate images\n(21 removed, 3,649 kept)", 194),
            ("4  Stratified split: 70 % training, 15 % validation, 15 % test", 268),
            ("5  Pre-process: resize and normalise all images; augment\nthe training images only", 330)]
    prev = 52
    for label, y in rows:
        h = 58 if "\n" in label else 44
        b += arrow(cx, prev, cx, y - 2)
        b += box(cx - w / 2, y, w, h, label, size=12.5)
        prev = y + h
    b += arrow(cx, prev, cx, 412)
    b += merge(cx, 428)
    b += arrow(cx, 444, cx, 462)
    loop = ["6  Take the next candidate model (CNN without augmentation,\nCNN with augmentation, MobileNetV3)",
            "7  Select the learning rate by short trial runs (validation F1)",
            "8  Train; after each epoch keep the weights with the best\nvalidation F1 (early stopping)",
            "9  Measure accuracy, macro F1, confusion matrix, size, latency"]
    y = 464
    for i, label in enumerate(loop):
        h = 58 if "\n" in label else 44
        b += box(cx - w / 2, y, w, h, label, size=12.5)
        b += arrow(cx, y + h, cx, y + h + 18)
        y += h + 20
    b += diamond(cx, y + 26, "more\ncandidates?", w=150, h=56)
    b += poly([(cx - 75, y + 26), (70, y + 26), (70, 428), (cx - 18, 428)], "[yes]", lx=80, ly=y + 14)
    b += arrow(cx, y + 54, cx, y + 78, "[no]", lx=cx + 10, ly=y + 66, anchor="start")
    y += 80
    b += box(cx - w / 2, y, w, 44, "10  Choose the model with the highest validation F1 (MobileNetV3)", size=12.5)
    b += arrow(cx, y + 44, cx, y + 62)
    b += box(cx - w / 2, y + 64, w, 44, "11  Report its test metrics and save weights + class list", size=12.5)
    b += arrow(cx, y + 108, cx, y + 126) + end(cx, y + 138)
    b += note(650, 150, 250, 130, "Cleaning precedes splitting so\nthat near-duplicates cannot fall\ninto both training and test sets.\nAugmentation is applied to\ntraining images only.")
    b += f'<line x1="650" y1="230" x2="{cx + w / 2}" y2="222" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 4"/>'
    svg("uml_activity_training", 920, y + 165, b, "")


# ---------------------------------------------------------------- 6. Sequence: one assistant request
def sequence():
    parts = [("Farm manager", 90), ("Web app", 300), ("Assistant", 510), ("Flower model\n(or Task 1 engine)", 730),
             ("Saved files", 940)]
    b = ""
    for name, x in parts:
        b += box(x - 85, 45, 170, 50, name, weight="bold", rx=4)
        b += f'<line x1="{x}" y1="95" x2="{x}" y2="700" stroke="#666666" stroke-width="1.3" stroke-dasharray="5 5"/>'
    X = {n.split("\n")[0]: x for n, x in parts}

    def msg(a, c, y, label, ret=False):
        x1, x2 = X[a], X[c]
        off = 6 if x2 > x1 else -6
        return arrow(x1 + off, y, x2 - off, y, label, dash=ret, ly=y - 10)

    b += msg("Farm manager", "Web app", 130, "1. upload photo, choose \"identify flower\"")
    b += msg("Web app", "Assistant", 175, "2. request (task, photo)")
    b += f'<rect x="504" y="185" width="12" height="440" fill="{FILL}" stroke="{INK}"/>'
    b += poly([(516, 205), (575, 205), (575, 228), (520, 228)], "3. is the task known and\nare the inputs complete?", lx=585, ly=216)
    b += f'<rect x="30" y="250" width="960" height="420" fill="none" stroke="{ACC}" stroke-width="1.6"/>'
    b += f'<rect x="30" y="250" width="50" height="20" fill="{ACC}"/>' + text(55, 260, "alt", 12, color="white", weight="bold")
    b += text(150, 282, "[request invalid]", 11.5, color=ACC)
    b += msg("Assistant", "Web app", 290, "4a. clear error message", ret=True)
    b += msg("Web app", "Farm manager", 320, "5a. show the message (no crash)", ret=True)
    b += f'<line x1="30" y1="335" x2="990" y2="335" stroke="{ACC}" stroke-dasharray="6 4"/>'
    b += text(150, 352, "[request valid]", 11.5, color=ACC)
    b += msg("Assistant", "Flower model", 385, "4. classify(photo)")
    b += msg("Flower model", "Saved files", 425, "5. load trained model (first time only)")
    b += msg("Saved files", "Flower model", 455, "model + class names", ret=True)
    b += poly([(736, 485), (800, 485), (800, 510), (740, 510)], "6. prepare photo exactly as\nin training, then predict", lx=810, ly=497)
    b += msg("Flower model", "Assistant", 550, "7. label, confidence, top 3", ret=True)
    b += msg("Assistant", "Web app", 595, "8. result: pictures + numbers + explanation", ret=True)
    b += msg("Web app", "Farm manager", 640, "9. display result", ret=True)
    svg("uml_sequence", 1030, 715, b, "UML sequence diagram: one request travelling through the system (Task 3)")


# ---------------------------------------------------------------- 7. Routing (activity with decision)
def routing():
    # UML activity diagram: decision node with guards, merge node, single final node.
    b = start(80, 210) + arrow(90, 210, 128, 210)
    b += box(130, 180, 170, 60, "Read the selected\ntask and image(s)")
    b += arrow(300, 210, 338, 210)
    b += diamond(420, 210, "valid task and\ninputs?", w=160, h=70)
    rows = [("[undistort]", "Calibration engine:\nremove lens distortion", 70),
            ("[denoise]", "Classical engine:\nbox / Gaussian / median", 130),
            ("[frequency]", "Classical engine:\nFFT low-pass / high-pass", 190),
            ("[edges]", "Classical engine:\nSobel + Canny", 250),
            ("[match]", "Classical engine:\nfeatures + ratio test + RANSAC", 310),
            ("[classify]", "Recognition engine:\nlabel + confidence", 370)]
    for label, target, y in rows:
        b += poly([(500, 210), (540, 210), (540, y), (690, y)], label, lx=550, ly=y - 11)
        b += box(692, y - 25, 260, 50, target, size=11.5)
        b += arrow(952, y, 990, y, head=False)
    b += f'<line x1="990" y1="70" x2="990" y2="370" stroke="{INK}" stroke-width="1.5"/>'
    b += arrow(990, 220, 1012, 220)
    b += merge(1030, 220, 18)
    b += arrow(1048, 220, 1068, 220)
    b += box(1070, 180, 150, 80, "Return a uniform\nresult: images,\nmetrics, message", fill=FILL, size=11.5)
    b += poly([(420, 245), (420, 420), (690, 420)], "[invalid: unknown task or missing image]", lx=430, ly=408)
    b += box(692, 400, 260, 42, "Return an explanatory error", fill=SOFT, size=11.5)
    b += arrow(1145, 260, 1145, 400)
    b += poly([(952, 421), (1130, 421)])
    b += end(1145, 414)
    svg("uml_routing", 1240, 460, b, "")


if __name__ == "__main__":
    for make in (use_case, components, activity_calibration, activity_classical, activity_training, sequence, routing):
        make()
    print("diagrams written to", OUT)
