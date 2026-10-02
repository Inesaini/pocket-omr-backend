"""Generate a documentation PDF for Pocket OMR (architecture, models, APIs, flows).

Renders diagrams with matplotlib and assembles the PDF with reportlab. Run with
the backend venv:  ./.venv/bin/python scripts/build_docs_pdf.py
Output: Pocket_OMR_Documentation.pdf at the repo root.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo/new-backend
ROOT = os.path.dirname(OUT_DIR)
IMG = os.path.join(OUT_DIR, "_docimg")
os.makedirs(IMG, exist_ok=True)

NAVY = "#053B76"
BLUE = "#0B96D9"
LIGHT = "#E6F4FB"
GREY = "#6b7280"
GREEN = "#1f9d55"
ORANGE = "#e08a00"

# --------------------------------------------------------------------------- #
# Diagram helpers (matplotlib)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=NAVY, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, weight="bold" if bold else "normal", wrap=True)


def _arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def _canvas(w=12, h=7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def diagram_architecture():
    fig, ax = _canvas(12, 7)
    ax.text(50, 97, "System Architecture", ha="center", fontsize=14, weight="bold", color=NAVY)
    # Clients
    _box(ax, 6, 78, 34, 12, "Web app (React 19 + Vite)\nTeacher: create exams, print PDFs, export grades",
         fc="#fff", ec=BLUE, tc=NAVY, fs=9, bold=True)
    _box(ax, 60, 78, 34, 12, "Mobile app (Flutter)\nScan sheets, grade, review, delete papers",
         fc="#fff", ec=BLUE, tc=NAVY, fs=9, bold=True)
    # API
    _box(ax, 22, 55, 56, 13, "FastAPI backend  (async)\nendpoints -> services -> models -> db",
         fc=LIGHT, ec=NAVY, tc=NAVY, fs=10, bold=True)
    # Subsystems
    _box(ax, 4, 33, 26, 12, "PostgreSQL\n(users, exams, questions,\nstudents, submissions)", fc="#fff", fs=8)
    _box(ax, 37, 33, 26, 12, "MinIO object store\n(scanned sheet images)", fc="#fff", fs=8)
    _box(ax, 70, 33, 26, 12, "Recognition models\nYOLOv8 + CNNs (TorchScript)\nOpenCV segmentation", fc="#fff", fs=8)
    # Recognition detail
    _box(ax, 22, 10, 56, 13, "OMR pipeline (app/recognition)\nsegment -> name CNN + bubble YOLO/CNN -> fuzzy match",
         fc="#FFF6E9", ec=ORANGE, tc=NAVY, fs=9, bold=True)
    # arrows
    _arrow(ax, (23, 78), (40, 68), BLUE)
    _arrow(ax, (77, 78), (60, 68), BLUE)
    _arrow(ax, (40, 55), (20, 45), NAVY)
    _arrow(ax, (50, 55), (50, 45), NAVY)
    _arrow(ax, (62, 55), (80, 45), NAVY)
    _arrow(ax, (83, 33), (60, 23), ORANGE)
    _arrow(ax, (50, 33), (50, 23), GREY, ls="--")
    return _save(fig, "architecture.png")


def diagram_er():
    fig, ax = _canvas(12, 8.2)
    ax.text(50, 99, "Data Model (Entity-Relationship)", ha="center", fontsize=14, weight="bold", color=NAVY)

    def ent(x, y, w, h, title, fields):
        _box(ax, x, y + h - 6, w, 6, title, fc=NAVY, ec=NAVY, tc="white", fs=9, bold=True)
        ax.add_patch(FancyBboxPatch((x, y), w, h - 6, boxstyle="square,pad=0",
                                    linewidth=1.2, edgecolor=NAVY, facecolor="#fff"))
        ax.text(x + 1.5, y + h - 7.5, fields, ha="left", va="top", fontsize=6.6, color="#222")

    ent(3, 70, 26, 24, "User",
        "id (PK)\nemail (unique)\nhashed_password\nfirst_name / last_name\nrole, is_active\ncreated/updated_at")
    ent(3, 44, 26, 18, "RefreshToken",
        "id (PK)\nuser_id (FK)\ntoken_hash\nexpires_at")
    ent(3, 22, 26, 16, "PasswordResetCode",
        "id (PK)\nuser_id (FK)\ncode, used\nexpires_at")
    ent(3, 4, 26, 13, "TokenBlacklist",
        "id (PK)\njti (unique)\nexpires_at")

    ent(37, 62, 30, 32, "Exam",
        "id (PK), user_id (FK)\ntitle, module, university\ndepartment, exam_date\nnum_questions\nchoices_per_question\nquestions_per_page\ncheckbox_type, grid_layout\nstatus, total_students\ncorrected_count\navg_confidence")
    ent(37, 36, 30, 20, "Question",
        "id (PK), exam_id (FK)\norder_index, text\ncorrect_answer\ncorrect_answers (JSON)\npoints")
    ent(37, 16, 30, 15, "Choice",
        "id (PK)\nquestion_id (FK)\norder_index, text")

    ent(72, 60, 26, 22, "ExamStudent (roster)",
        "id (PK), exam_id (FK)\nfirst_name, last_name\ngroup_name\nregistration_number")
    ent(72, 26, 26, 30, "StudentSubmission",
        "id (PK), exam_id (FK)\nfirst_name, last_name\nstudent_id\nscore / max_score\nconfidence, status\nneeds_review\nflagged_questions (JSON)\nsheet_image_path\nrecognized_name")

    # relationships (crow's-foot-ish: label 1..*)
    def rel(p1, p2, label):
        _arrow(ax, p1, p2, NAVY, style="-|>")
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 1.5, label, ha="center", fontsize=6.5, color=GREEN, weight="bold")

    rel((29, 84), (37, 80), "1..*")        # User -> Exam
    rel((16, 70), (16, 62), "1..*")        # User -> RefreshToken
    rel((16, 44), (16, 38), "1..*")        # User -> PasswordResetCode
    rel((52, 62), (52, 56), "1..*")        # Exam -> Question
    rel((52, 36), (52, 31), "1..*")        # Question -> Choice
    rel((67, 74), (72, 70), "1..*")        # Exam -> ExamStudent
    rel((67, 68), (72, 50), "1..*")        # Exam -> StudentSubmission
    ax.text(85, 20, "fuzzy name match\n(roster <-> submission)", ha="center", fontsize=6.4,
            color=ORANGE, style="italic")
    _arrow(ax, (85, 26), (85, 24.5), ORANGE, ls="--")
    return _save(fig, "er.png")


def diagram_pipeline():
    fig, ax = _canvas(12, 7.6)
    ax.text(50, 98, "Grading Pipeline (one scanned sheet)", ha="center", fontsize=14, weight="bold", color=NAVY)
    steps = [
        (4, 80, "1. Upload\nimage / .zip", LIGHT),
        (4, 60, "2. Segmentation precheck\n(reject unreadable)", LIGHT),
        (4, 40, "3. Store image\n(MinIO)", "#fff"),
        (38, 80, "4. Segment sheet\nprocess_sheet():\norient (QR/markers),\nwarp, crop regions", "#FFF6E9"),
        (38, 56, "5a. Name region\nbox detect -> char CNN\n-> predicted name", "#fff"),
        (38, 30, "5b. Answers region\nYOLO detect -> assign grid\n-> CNN + ink net", "#fff"),
        (72, 56, "6. Fuzzy match name\nto roster (<=0.45)\nelse keep recognised", "#fff"),
        (72, 30, "7. Score: exact-set\nmatch x points\n+ review flags", "#fff"),
        (72, 6, "8. Persist\nStudentSubmission\n+ recompute totals", LIGHT),
    ]
    coords = {}
    for x, y, t, fc in steps:
        _box(ax, x, y, 24, 16, t, fc=fc, fs=8)
        coords[t[:2]] = (x, y)
    _arrow(ax, (16, 80), (16, 76), NAVY)
    _arrow(ax, (16, 60), (16, 56), NAVY)
    _arrow(ax, (28, 68), (38, 84), NAVY)   # upload chain -> segment
    _arrow(ax, (50, 80), (50, 72), NAVY)   # segment -> name
    _arrow(ax, (50, 56), (50, 46), NAVY)   # segment -> answers
    _arrow(ax, (62, 60), (72, 62), NAVY)   # name -> match
    _arrow(ax, (62, 36), (72, 38), NAVY)   # answers -> score
    _arrow(ax, (84, 56), (84, 46), NAVY)   # match -> persist (via)
    _arrow(ax, (84, 30), (84, 22), NAVY)   # score -> persist
    ax.text(50, 2, "Any failure -> sheet kept PENDING and re-gradable later (regrade endpoint).",
            ha="center", fontsize=8, color=GREY, style="italic")
    return _save(fig, "pipeline.png")


def diagram_layers():
    fig, ax = _canvas(11, 5.6)
    ax.text(50, 95, "Backend Request Flow (layered)", ha="center", fontsize=14, weight="bold", color=NAVY)
    layers = [
        (8, "HTTP request (Bearer JWT)", "#fff"),
        (8, "", ""),
    ]
    boxes = [
        (4, 70, "API endpoint\n(app/api/v1/endpoints)\nauth deps, validation", LIGHT),
        (28, 70, "Pydantic schemas\n(request/response DTOs)", "#fff"),
        (52, 70, "Service layer\n(app/services)\nbusiness logic", LIGHT),
        (76, 70, "Recognition seam\n(grade_sheet)", "#FFF6E9"),
        (28, 40, "SQLAlchemy models\n(async ORM)", LIGHT),
        (52, 40, "PostgreSQL", "#fff"),
        (76, 40, "MinIO storage", "#fff"),
    ]
    for x, y, t, fc in boxes:
        _box(ax, x, y, 20, 16, t, fc=fc, fs=8)
    _arrow(ax, (24, 78), (28, 78), NAVY)
    _arrow(ax, (48, 78), (52, 78), NAVY)
    _arrow(ax, (72, 78), (76, 78), ORANGE)
    _arrow(ax, (62, 70), (50, 56), NAVY)   # service -> models
    _arrow(ax, (48, 48), (52, 48), NAVY)   # models -> pg
    _arrow(ax, (62, 70), (84, 56), NAVY)   # service -> minio
    ax.text(50, 90, "endpoints  ->  services  ->  models / storage  ->  DB",
            ha="center", fontsize=9, color=GREY)
    return _save(fig, "layers.png")


def diagram_auth():
    fig, ax = _canvas(11, 4.8)
    ax.text(50, 95, "Authentication (JWT) Flow", ha="center", fontsize=14, weight="bold", color=NAVY)
    _box(ax, 3, 55, 22, 16, "register / login\n-> access (15m)\n+ refresh (7d)", LIGHT, fs=8)
    _box(ax, 30, 55, 22, 16, "API call with\nAuthorization:\nBearer <access>", "#fff", fs=8)
    _box(ax, 57, 55, 18, 16, "401 expired?", "#FFF6E9", fs=8)
    _box(ax, 80, 55, 17, 16, "POST /refresh\nrotate tokens", LIGHT, fs=8)
    _box(ax, 30, 22, 22, 16, "logout ->\nblacklist jti", "#fff", fs=8)
    _arrow(ax, (25, 63), (30, 63), NAVY)
    _arrow(ax, (52, 63), (57, 63), NAVY)
    _arrow(ax, (75, 63), (80, 63), ORANGE)
    _arrow(ax, (88, 55), (45, 38), GREY, ls="--")
    _arrow(ax, (41, 55), (41, 38), NAVY)
    ax.text(50, 8, "Passwords: bcrypt, min 8 chars w/ upper+lower+digit. Refresh tokens hashed in DB.",
            ha="center", fontsize=8, color=GREY, style="italic")
    return _save(fig, "auth.png")


def _onehot(ax, x, y, vec, cell=4.2, gap=0.7):
    """Draw a one-hot row of cells (filled cells shaded)."""
    for i, v in enumerate(vec):
        cx = x + i * (cell + gap)
        ax.add_patch(Rectangle((cx, y), cell, cell, lw=1.1, edgecolor=NAVY,
                               facecolor=(BLUE if v else "white")))
        ax.text(cx + cell / 2, y + cell / 2, str(v), ha="center", va="center",
                fontsize=7.5, weight="bold", color=("white" if v else NAVY))
    return x + len(vec) * (cell + gap)


def diagram_scoring():
    fig, ax = _canvas(12, 7.4)
    ax.text(50, 99, "Correction: one-hot encoding -> comparison",
            ha="center", fontsize=14, weight="bold", color=NAVY)

    # ---- top pipeline ----
    _box(ax, 2, 84, 20, 11, "Teacher marks\ncorrect choice(s)\ne.g. Q = B", LIGHT, fs=8)
    _box(ax, 27, 84, 23, 11, "Encode over choices\n[A B C D]", "#fff", fs=8)
    _box(ax, 55, 84, 20, 11, "Student's detected\nfilled bubbles", LIGHT, fs=8)
    _box(ax, 80, 84, 18, 11, "Compare\n(set / vector ==)", "#FFF6E9", ec=ORANGE, fs=8)
    _arrow(ax, (22, 89.5), (27, 89.5), NAVY)
    _arrow(ax, (50, 89.5), (55, 89.5), NAVY)
    _arrow(ax, (75, 89.5), (80, 89.5), ORANGE)
    # one-hot under "encode"
    ax.text(38.5, 81, "key one-hot = [0,1,0,0]  =  set {1}", ha="center", fontsize=7, color=GREY)
    ax.text(65, 81, "student = [0,1,0,0]  =  {1}", ha="center", fontsize=7, color=GREY)
    ax.text(89, 80.5, "equal -> +points\nelse 0", ha="center", fontsize=7, color=ORANGE)

    # ---- worked example table ----
    ax.text(50, 73, "Worked example  (choices A B C D, 0-based index)", ha="center",
            fontsize=10, weight="bold", color=NAVY)
    CELL, GAPC = 3.0, 0.5
    XQ, XKEY, XKV, XSTUD, XSV, XEQ, XPTS = 3, 9, 25, 42, 60, 79, 85
    head_y = 67
    for label, x in [("Q", XQ), ("Correct", XKEY), ("key one-hot", XKV),
                     ("Student", XSTUD), ("student one-hot", XSV), ("=?", XEQ), ("pts", XPTS)]:
        ax.text(x, head_y, label, ha="left", fontsize=7.5, weight="bold", color=BLUE)
    rows = [
        ("Q1", "B {1}", [0, 1, 0, 0], "B {1}", [0, 1, 0, 0], True, "1 / 1"),
        ("Q2", "C {2}", [0, 0, 1, 0], "B {1}", [0, 1, 0, 0], False, "0 / 1"),
        ("Q3", "A {0}", [1, 0, 0, 0], "A {0}", [1, 0, 0, 0], True, "1 / 1"),
        ("Q4", "B,D {1,3}", [0, 1, 0, 1], "B {1}", [0, 1, 0, 0], False, "0 / 2"),
        ("Q5", "C {2}", [0, 0, 1, 0], "blank {}", [0, 0, 0, 0], False, "0 / 1"),
    ]
    y = 61
    for q, key, kv, stud, sv, ok, pts in rows:
        ax.text(XQ, y + 1.5, q, ha="left", fontsize=8, color="#222")
        ax.text(XKEY, y + 1.5, key, ha="left", fontsize=7, color="#222")
        _onehot(ax, XKV, y, kv, cell=CELL, gap=GAPC)
        ax.text(XSTUD, y + 1.5, stud, ha="left", fontsize=7, color="#222")
        _onehot(ax, XSV, y, sv, cell=CELL, gap=GAPC)
        ax.text(XEQ, y + 1.5, "=" if ok else "≠", ha="center", fontsize=12,
                color=(GREEN if ok else "#c0392b"), weight="bold")
        ax.text(XPTS, y + 1.5, pts, ha="left", fontsize=8.5,
                color=(GREEN if ok else "#c0392b"), weight="bold")
        y -= 8

    ax.text(50, 13, "Full marks for a question only when the two are EXACTLY equal "
            "(all correct marked, nothing extra) — no partial credit.",
            ha="center", fontsize=8.5, color=NAVY, style="italic")
    ax.text(50, 6, "grade = Σ points(matched)  /  Σ points(keyed)   =   2 / 6",
            ha="center", fontsize=10, weight="bold", color=NAVY)
    return _save(fig, "scoring.png")


# --------------------------------------------------------------------------- #
# PDF assembly (reportlab)
# --------------------------------------------------------------------------- #
def build_pdf():
    arch = diagram_architecture()
    er = diagram_er()
    pipe = diagram_pipeline()
    layers = diagram_layers()
    auth = diagram_auth()
    scoring = diagram_scoring()

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor(NAVY), fontSize=18, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor(BLUE), fontSize=13, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.3, leading=13, alignment=TA_LEFT)
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor(GREY))
    cap = ParagraphStyle("Cap", parent=small, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)

    doc = SimpleDocTemplate(os.path.join(ROOT, "Pocket_OMR_Documentation.pdf"),
                            pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Pocket OMR - Technical Documentation")
    E = []

    def img(path, w=170):
        from PIL import Image as PILImage
        iw, ih = PILImage.open(path).size
        wmm = w * mm
        return Image(path, width=wmm, height=wmm * ih / iw)

    def para(t, s=body):
        E.append(Paragraph(t, s))

    def table(headers, rows, colw):
        data = [[Paragraph(f"<b>{h}</b>", small) for h in headers]]
        for r in rows:
            data.append([Paragraph(str(c), small) for c in r])
        t = Table(data, colWidths=[c * mm for c in colw], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f7fb")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        E.append(t)

    # ---- Cover ----
    E.append(Spacer(1, 60 * mm))
    para("Pocket OMR", ParagraphStyle("T", parent=h1, fontSize=34, alignment=TA_CENTER))
    para("Technical Documentation", ParagraphStyle("T2", parent=h2, fontSize=16, alignment=TA_CENTER, textColor=colors.HexColor(GREY)))
    E.append(Spacer(1, 8 * mm))
    para("Architecture, data model, API reference, and the logic behind the "
         "web app, mobile app, backend, and OMR grading engine.",
         ParagraphStyle("T3", parent=body, alignment=TA_CENTER, fontSize=11))
    E.append(PageBreak())

    # ---- 1. Overview ----
    para("1. Overview", h1)
    para("Pocket OMR is a full-stack system for creating, printing, and grading "
         "optical-mark-recognition (OMR) multiple-choice exams. A teacher builds an "
         "exam in the <b>web app</b>, prints the generated answer sheets, and after the "
         "exam scans the filled sheets with the <b>mobile app</b>. A <b>FastAPI backend</b> "
         "stores everything, runs the recognition pipeline (handwritten name + filled "
         "bubbles), matches each sheet to the student roster, scores it against the "
         "answer key, and returns results that the apps display and export.")
    para("Core capabilities", h2)
    para("&bull; Build exams: title/metadata, questions, choices, multiple-correct "
         "answers, per-question points, double-column or linear grids.<br/>"
         "&bull; Generate printable PDFs: question sheet, grid (answer) sheet, correction sheet – each with a QR code.<br/>"
         "&bull; Upload a roster (Excel/CSV) of students per exam.<br/>"
         "&bull; Scan sheets (camera or .zip) and auto-grade: name recognition + bubble grading.<br/>"
         "&bull; Fuzzy-match recognised names to the roster; flag uncertain ones for review.<br/>"
         "&bull; Per-student scores, model confidence, review flags; delete individual papers.<br/>"
         "&bull; Export a grades spreadsheet (name, group, registration number, score, %).")

    para("2. System Architecture", h1)
    E.append(img(arch, 165))
    para("Two clients talk to one FastAPI backend over HTTP/JSON. The backend "
         "persists structured data in PostgreSQL, stores scanned images in MinIO, "
         "and runs the OMR recognition models (OpenCV + YOLOv8 + TorchScript CNNs).", small)
    E.append(PageBreak())

    # ---- 3. Tech stack ----
    para("3. Technology Stack", h1)
    table(["Layer", "Technology"], [
        ["Web app", "React 19, Vite, React Router 7, Tailwind, Context API"],
        ["Mobile app", "Flutter (provider + Dio), camera, image picker"],
        ["Backend", "FastAPI, async SQLAlchemy, Pydantic, Alembic"],
        ["Database", "PostgreSQL (async via asyncpg)"],
        ["Object storage", "MinIO (scanned sheet images)"],
        ["PDF / QR", "ReportLab + qrcode"],
        ["Recognition", "OpenCV, Ultralytics YOLOv8, PyTorch TorchScript CNNs, openpyxl"],
        ["Auth", "JWT (HS256) access + refresh, bcrypt password hashing"],
    ], [32, 138])

    para("4. Backend Architecture (layered)", h1)
    E.append(img(layers, 165))
    para("Requests flow endpoints → services → models/storage. Endpoints handle "
         "HTTP, auth dependencies and validation; services hold business logic; the "
         "<i>recognition seam</i> (grade_sheet) is the single integration point for the "
         "OMR models, so the models can change without touching the rest of the system.", small)
    E.append(PageBreak())

    # ---- 5. Data model ----
    para("5. Data Model", h1)
    E.append(img(er, 168))
    E.append(Spacer(1, 4))
    para("Key relationships", h2)
    table(["Relationship", "Meaning"], [
        ["User 1–* Exam", "A teacher owns many exams (cascade delete)."],
        ["Exam 1–* Question 1–* Choice", "An exam has ordered questions; each question has ordered choices."],
        ["Exam 1–* ExamStudent", "The uploaded roster for that exam."],
        ["Exam 1–* StudentSubmission", "One row per scanned/graded sheet."],
        ["User 1–* RefreshToken / PasswordResetCode", "Auth artefacts, hashed / single-use."],
        ["Submission ↔ ExamStudent", "Linked at grade time by fuzzy name match (not a FK)."],
    ], [55, 115])
    E.append(PageBreak())

    # ---- 6. API reference ----
    para("6. API Reference", h1)
    para("Auth — prefix <font face='Courier'>/api/v1/auth</font>", h2)
    table(["Method", "Path", "Purpose"], [
        ["POST", "/register", "Create teacher account → tokens + user"],
        ["POST", "/login", "Authenticate → access + refresh tokens"],
        ["GET", "/me", "Current user (Bearer)"],
        ["PUT", "/me", "Update profile"],
        ["POST", "/refresh", "Rotate access/refresh tokens"],
        ["POST", "/logout", "Blacklist the token (jti)"],
        ["POST", "/send-reset-code", "Email a password-reset code"],
        ["POST", "/verify-reset-code", "Verify the reset code"],
        ["POST", "/reset-password", "Set a new password"],
    ], [18, 44, 108])

    para("Exams — prefix <font face='Courier'>/api/v1/exams</font>", h2)
    table(["Method", "Path", "Purpose"], [
        ["POST", "", "Create exam (form + questions + roster)"],
        ["GET", "", "List the user's exams"],
        ["GET", "/recent", "Recent activity (by last update)"],
        ["GET", "/to-correct", "Exams available to correct"],
        ["GET", "/history", "Graded history with stats"],
        ["GET", "/{id}", "Full exam"],
        ["PUT", "/{id}", "Update exam"],
        ["DELETE", "/{id}", "Delete exam"],
        ["GET", "/{id}/mobile", "Mobile view: results + aggregates"],
        ["POST", "/{id}/upload-images", "Upload sheets (image or .zip) → grade each"],
        ["GET", "/{id}/results.xlsx", "Download grades spreadsheet"],
        ["DELETE", "/{id}/submissions/{sid}", "Delete one graded paper"],
        ["POST", "/{id}/regrade", "Re-run grading on pending sheets"],
    ], [18, 58, 94])

    para("Other routers", h2)
    table(["Method", "Path", "Purpose"], [
        ["POST", "/api/v1/students/parse-file", "Parse an uploaded roster (Excel/CSV)"],
        ["POST", "/generate-pdf", "Render question / grid / correction sheet PDF"],
        ["POST", "/grade", "Stateless grade of an uploaded sheet"],
        ["POST", "/roster", "Stateless roster parse for matching"],
        ["GET", "/health, /health/db", "Liveness and DB connectivity"],
    ], [18, 60, 92])
    E.append(PageBreak())

    # ---- 7. Auth ----
    para("7. Authentication & Security", h1)
    E.append(img(auth, 160))
    para("JWT HS256: short-lived access token (15 min) + long-lived refresh token "
         "(7 days). The web client auto-refreshes on a 401 and redirects to sign-in "
         "if the session is truly dead. Refresh tokens are stored hashed; logout "
         "blacklists the token's <font face='Courier'>jti</font>. Passwords are bcrypt-hashed and must be "
         "8+ chars with upper, lower and digit.", small)

    # ---- 8. Exam creation ----
    para("8. Exam Creation & PDF Generation (web)", h1)
    para("The web app is a 5-step wizard: <b>ExamConfig → QuestionsCreation → "
         "QuestionSheet → GridSheet → CorrectionSheet</b>. State lives in React "
         "Contexts (ExamConfig, Questions, ExamList). Exams are persisted entirely in "
         "the backend (no localStorage except auth tokens). The teacher's chosen "
         "correct answer(s) become a one-hot answer key used later for grading by "
         "comparison. PDFs are produced by the backend <font face='Courier'>/generate-pdf</font> endpoint and "
         "each sheet carries a QR code with exam metadata (title, module, #questions, #choices).")

    para("9. The Grading Pipeline", h1)
    E.append(img(pipe, 165))
    para("grade_sheet() orchestrates the per-sheet flow. Unreadable sheets are "
         "rejected up front (returned as failedSheets); anything the models can't "
         "handle is stored PENDING and re-gradable later.", small)
    E.append(PageBreak())

    # ---- 10. Recognition detail ----
    para("10. Recognition Engine (details)", h1)
    para("10.1 Segmentation", h2)
    para("<font face='Courier'>process_sheet()</font> loads the photo (honouring EXIF), finds the page, "
         "resolves orientation (decoded QR first, then corner-marker layout), warps "
         "to a canonical rectangle, and crops the <i>personal-info</i> and <i>answers</i> "
         "regions. Orientation only trusts a QR that actually decodes – a located-but-"
         "undecoded QR is ignored (it had caused 90° mis-rotations).")
    para("10.2 Name recognition + roster matching", h2)
    para("The personal-info region's character cells are detected as the interior "
         "holes of the printed grid (robust to connected tables), each cell is "
         "classified by a TorchScript char CNN, and the four fields (first/last name, "
         "group, registration number) are reassembled. The result is fuzzy-matched to "
         "the exam roster with a weighted normalised Levenshtein distance "
         "(registration number 0.45, names 0.25 each, group 0.05). If the best match "
         "is worse than 0.45 the sheet is <i>not</i> attributed to the nearest student – the "
         "recognised name is kept and the sheet flagged for review.")
    para("10.3 Bubble grading", h2)
    para("YOLOv8 detects bubbles on the answers crop; <font face='Courier'>assign_grid</font> lays them into a "
         "(question, choice) grid – splitting double columns, assigning the choice by "
         "the bubble's rank within its row (immune to tilt), using ceil(n/columns) so "
         "odd counts split correctly, pruning phantom rows from stray specks, and "
         "capping each row to its choice count. A CNN classifies each bubble filled/"
         "empty; an <b>ink safety net</b> rescues faint pen/pencil fills the CNN misses by "
         "comparing each bubble's interior-vs-border contrast (shadow-invariant). "
         "Scoring is an exact-set match per question (the student must mark exactly "
         "the correct choice set) weighted by the question's points.")
    para("10.4 Confidence & review flags", h2)
    para("Per bubble the CNN gives a filled probability p; its certainty is "
         "max(p, 1−p). A sheet's <b>confidence</b> is the average certainty over its "
         "bubbles; a question's confidence is its least-sure bubble. A question is "
         "<b>flagged for review</b> when the model is unsure, a bubble is missing, the "
         "student double-marked, or the answer was recovered by the ink net. "
         "Confidence measures mark clarity – not whether the score is correct.")
    E.append(PageBreak())

    # ---- 11. Mobile ----
    para("11. Mobile App Flow", h1)
    para("A bottom-tab shell (Profile, Home, Exam list, History). <b>Home</b> shows recent "
         "activity; <b>Exam list</b> shows exams to correct; <b>History</b> shows graded exams "
         "with photos-processed and average-confidence stats. Grading flow: pick an "
         "exam → capture sheets (camera) or upload a .zip → backend grades each → "
         "<b>results overview</b> shows every paper's score, confidence and review flag. "
         "A paper can be deleted (swipe), and segmentation failures raise a popup so "
         "those sheets can be re-shot. Screens re-fetch when their tab is reselected so "
         "newly graded papers appear immediately.")

    para("12. Correction: From Answer Key to Comparison", h1)
    para("Grading is a per-question <b>set / one-hot comparison</b> between the teacher's "
         "answer key and the student's detected marks — completely separate from the "
         "extraction (which only produces the student's filled set).", body)
    E.append(img(scoring, 165))
    E.append(Spacer(1, 4))

    para("12.1 Encoding the right answers", h2)
    para("When the teacher selects the correct choice(s) for a question, each choice is "
         "an index over the option list <font face='Courier'>[A, B, C, D] = [0, 1, 2, 3]</font>. "
         "The selection is stored as a list of correct indices, "
         "<font face='Courier'>correct_answers</font> (e.g. <font face='Courier'>[1]</font> for B, or "
         "<font face='Courier'>[1, 3]</font> for B and D). Equivalently this is a <b>one-hot vector</b> "
         "over the choices: B → <font face='Courier'>[0,1,0,0]</font>, B&amp;D → "
         "<font face='Courier'>[0,1,0,1]</font>. The web app builds exactly this one-hot answer key "
         "(<font face='Courier'>answerVector.js</font>); the backend stores it as the index set. The two "
         "representations are equivalent — a set of indices and its one-hot vector carry "
         "the same information.")

    para("12.2 Encoding the student's answer", h2)
    para("For each question the bubble pipeline outputs the <b>set of choice indices it "
         "found filled</b> (e.g. <font face='Courier'>{1}</font> for a single B, "
         "<font face='Courier'>{}</font> for blank, <font face='Courier'>{1,2}</font> for a double-mark). "
         "This is the same encoding as the key, so they can be compared directly.")

    para("12.3 The comparison", h2)
    para("Question <i>i</i> on the sheet is compared to the <i>i</i>-th question of the exam "
         "(ordered by <font face='Courier'>order_index</font>). The question earns its points "
         "<b>only when the two sets are exactly equal</b>:")
    para("<font face='Courier'>award points[i]  ⇔  set(student_filled[i]) == set(correct_answers[i])</font>",
         ParagraphStyle("Code", parent=body, fontName="Courier", fontSize=9,
                        backColor=colors.HexColor("#f1f7fb"), borderPadding=4, leading=14))
    para("Exact equality means the student must have marked <b>every</b> correct choice and "
         "<b>nothing else</b> — so a partial answer on a multi-correct question, a wrong "
         "choice, a blank, or a double-mark all score 0 for that question. There is no "
         "partial credit.")

    para("12.4 Aggregating into a grade", h2)
    para("<font face='Courier'>max_score = Σ points</font> over questions that have a defined correct "
         "answer; <font face='Courier'>score = Σ points</font> over the questions whose sets matched. The "
         "paper's grade is <font face='Courier'>score / max_score</font>. Questions with no defined "
         "correct answer are excluded from both. Review flags and confidence are computed "
         "separately and never change the score.")

    para("Summary of the rules", h2)
    table(["Rule", "Behaviour"], [
        ["Encoding", "Correct answers and student marks are both sets of 0-based choice indices (one-hot equivalent)."],
        ["Per-question points", "Each question is worth its teacher-set points (default 1)."],
        ["Exact-set match", "Full points only if student set == correct set; otherwise 0 (no partial credit)."],
        ["Multiple correct", "Student must mark all correct choices and nothing else."],
        ["Blank / double-mark", "Empty or multi-marked sets that differ from the key score 0."],
        ["No-answer questions", "A question with no correct answer is excluded from score and max."],
        ["Grade", "score / max_score, where max = sum of points over keyed questions."],
        ["Review / confidence", "Computed separately; do not affect the score."],
    ], [42, 128])

    para("13. Known Limitations", h1)
    para("&bull; A question physically cropped out of frame (sheet shot rotated) cannot be recovered – reshoot flat.<br/>"
         "&bull; Very faint pencil marks approach the limit of distinguishable-from-blank; black pen grades most reliably.<br/>"
         "&bull; Auto name-matching needs the real student roster uploaded; otherwise names are recognised but flagged for manual assignment.<br/>"
         "&bull; The QR currently carries exam metadata (not the unique id); same-exam verification at upload is a proposed enhancement.", small)

    doc.build(E)
    print("WROTE", os.path.join(ROOT, "Pocket_OMR_Documentation.pdf"))


if __name__ == "__main__":
    build_pdf()
