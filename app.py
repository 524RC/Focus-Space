import base64
import io
import json
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from html import escape, unescape
from html.parser import HTMLParser

from flask import (
    Flask, Response, abort, g, jsonify, render_template, request,
    send_from_directory,
)
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    from pypdf import PdfReader      # reads text out of PDFs
except ImportError:
    PdfReader = None


# ======================================================================
# SETUP
# ======================================================================

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB per request

DB_PATH = "notes.db"

# Images and PDFs are stored as files, next to app.py, under random names.
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PDF_BYTES = 30 * 1024 * 1024
MAX_AI_FILE_BYTES = 18 * 1024 * 1024          # Gemini's limit for inline files

UPLOAD_NAME_RE = re.compile(r"^[0-9a-f]{32}\.(png|jpg|gif|webp|pdf)$")
UPLOAD_URL_RE = re.compile(r"^/uploads/[0-9a-f]{32}\.(?:png|jpg|gif|webp|pdf)$")
UPLOAD_FIND_RE = re.compile(r"/uploads/([0-9a-f]{32}\.(?:png|jpg|gif|webp|pdf))")
MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".pdf": "application/pdf",
}

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Tried in order. If one is overloaded, the next is used.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]


# ======================================================================
# DATABASE
# ======================================================================

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row

    return g.db


@app.teardown_appcontext
def close_db(error):
    db = g.pop("db", None)

    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS folders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'Untitled',
                content TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Upgrade older databases that don't have folder_id yet.
        columns = [row[1] for row in db.execute("PRAGMA table_info(notes)")]

        if "folder_id" not in columns:
            db.execute("ALTER TABLE notes ADD COLUMN folder_id INTEGER")

        # 0 = plain text (notes from older versions), 1 = HTML (rich text).
        if "is_html" not in columns:
            db.execute(
                "ALTER TABLE notes ADD COLUMN is_html INTEGER NOT NULL DEFAULT 0"
            )

        db.execute("""
            CREATE TABLE IF NOT EXISTS todos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                done INTEGER NOT NULL DEFAULT 0,
                due_date TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                author TEXT NOT NULL DEFAULT 'You',
                body TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'owner',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS shares (
                token TEXT PRIMARY KEY,
                note_id INTEGER,
                folder_id INTEGER,
                allow_comments INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.commit()


def now_string():
    # Same format SQLite's CURRENT_TIMESTAMP uses (UTC).
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ======================================================================
# RICH TEXT HELPERS
#
# Notes are stored as HTML (from the editor). Everything that comes in is
# run through an allow-list sanitizer, so shared pages and the editor can
# never be used to run scripts.
# ======================================================================

ALLOWED_TAGS = {
    "p", "br", "strong", "b", "em", "i", "u", "s", "strike",
    "h1", "h2", "h3", "ul", "ol", "li", "blockquote", "pre",
    "code", "a", "hr", "div", "span", "img",
}
VOID_TAGS = {"br", "hr", "img"}
SKIP_TAGS = {"script", "style", "title"}   # their contents are dropped


class Sanitizer(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.stack = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
            return

        if self.skip or tag not in ALLOWED_TAGS:
            return

        extra = ""

        if tag == "a":
            href = (dict(attrs).get("href") or "").strip()

            if (
                href.lower().startswith(("http://", "https://", "mailto:"))
                or UPLOAD_URL_RE.match(href)
            ):
                extra = (
                    f' href="{escape(href, quote=True)}"'
                    ' target="_blank" rel="noopener noreferrer"'
                )

        elif tag == "img":
            attributes = dict(attrs)
            src = (attributes.get("src") or "").strip()

            # Only pictures this app stored itself. Never outside images.
            if not UPLOAD_URL_RE.match(src) or src.endswith(".pdf"):
                return

            alt = (attributes.get("alt") or "")[:200]
            extra = (
                f' src="{escape(src, quote=True)}"'
                f' alt="{escape(alt, quote=True)}"'
            )

        self.out.append(f"<{tag}{extra}>")

        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return

        if self.skip or tag in VOID_TAGS or tag not in self.stack:
            return

        while self.stack:
            open_tag = self.stack.pop()
            self.out.append(f"</{open_tag}>")

            if open_tag == tag:
                break

    def handle_data(self, data):
        if not self.skip:
            self.out.append(escape(data))

    def result(self):
        while self.stack:
            self.out.append(f"</{self.stack.pop()}>")

        return "".join(self.out)


def clean_html(html):
    parser = Sanitizer()
    parser.feed(html or "")
    parser.close()
    return parser.result()


def text_to_html(text):
    text = (text or "").replace("\r\n", "\n").strip()

    if not text:
        return ""

    paragraphs = re.split(r"\n\s*\n", text)

    return "".join(
        "<p>" + escape(p.strip("\n")).replace("\n", "<br>") + "</p>"
        for p in paragraphs
    )


def content_html(row):
    """Safe HTML for a note row. Old plain-text notes are converted."""
    if row["is_html"]:
        return clean_html(row["content"])

    return text_to_html(row["content"])


class TextExtractor(HTMLParser):

    BLOCKS = {"p", "div", "br", "h1", "h2", "h3", "li",
              "blockquote", "pre", "ul", "ol", "hr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
            return

        if tag in self.BLOCKS:
            self.parts.append("\n")

        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
            return

        if tag in self.BLOCKS and tag != "br":
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

    def text(self):
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_text(html):
    parser = TextExtractor()
    parser.feed(html or "")
    parser.close()
    return parser.text()


def content_text(row):
    """Plain text version of a note row (AI, previews, .txt export)."""
    return html_to_text(content_html(row))


# ---------- Markdown -> HTML ----------

def inline_md(text):
    text = escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2">\1</a>',
        text,
    )
    return text


def md_to_html(text):
    lines = (text or "").replace("\r\n", "\n").split("\n")

    out = []
    para = []
    code = []
    state = {"list": None, "in_code": False}

    def flush_para():
        if para:
            out.append(
                "<p>" + "<br>".join(inline_md(line) for line in para) + "</p>"
            )
            para.clear()

    def close_list():
        if state["list"]:
            out.append(f"</{state['list']}>")
            state["list"] = None

    def open_list(kind):
        if state["list"] != kind:
            close_list()
            out.append(f"<{kind}>")
            state["list"] = kind

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if state["in_code"]:
                out.append("<pre>" + escape("\n".join(code), quote=False) + "</pre>")
                code.clear()
                state["in_code"] = False
            else:
                flush_para()
                close_list()
                state["in_code"] = True
            continue

        if state["in_code"]:
            code.append(line)
            continue

        if not stripped:
            flush_para()
            close_list()
            continue

        match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if match:
            flush_para()
            close_list()
            level = min(len(match.group(1)), 3)
            out.append(f"<h{level}>{inline_md(match.group(2))}</h{level}>")
            continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            flush_para()
            close_list()
            out.append("<hr>")
            continue

        match = re.match(r"^[-*+]\s+(.*)$", stripped)
        if match:
            flush_para()
            open_list("ul")
            out.append(f"<li>{inline_md(match.group(1))}</li>")
            continue

        match = re.match(r"^\d+[.)]\s+(.*)$", stripped)
        if match:
            flush_para()
            open_list("ol")
            out.append(f"<li>{inline_md(match.group(1))}</li>")
            continue

        match = re.match(r"^>\s?(.*)$", stripped)
        if match:
            flush_para()
            close_list()
            out.append(f"<blockquote>{inline_md(match.group(1))}</blockquote>")
            continue

        close_list()
        para.append(stripped)

    if state["in_code"]:
        out.append("<pre>" + escape("\n".join(code), quote=False) + "</pre>")

    flush_para()
    close_list()

    return "".join(out)


# ---------- HTML -> Markdown ----------

class MarkdownWriter(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.buf = ""
        self.lists = []       # stack of [kind, counter]
        self.href = None
        self.in_pre = False

    def block(self):
        if self.buf and not self.buf.endswith("\n\n"):
            self.buf = self.buf.rstrip("\n") + "\n\n"

    def line(self):
        if self.buf and not self.buf.endswith("\n"):
            self.buf += "\n"

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self.block()
            self.buf += "#" * int(tag[1]) + " "

        elif tag in ("p", "blockquote"):
            self.block()
            if tag == "blockquote":
                self.buf += "> "

        elif tag == "div":
            self.line()

        elif tag == "br":
            self.buf += "  \n"

        elif tag in ("strong", "b"):
            self.buf += "**"

        elif tag in ("em", "i"):
            self.buf += "*"

        elif tag in ("s", "strike"):
            self.buf += "~~"

        elif tag == "code" and not self.in_pre:
            self.buf += "`"

        elif tag == "pre":
            self.block()
            self.buf += "```\n"
            self.in_pre = True

        elif tag == "a":
            href = (dict(attrs).get("href") or "").strip()

            # Links to files stored in this app would be dead outside it.
            if href and not href.startswith("/uploads/"):
                self.href = href
                self.buf += "["

        elif tag in ("ul", "ol"):
            if self.lists:
                self.line()
            else:
                self.block()
            self.lists.append([tag, 0])

        elif tag == "li":
            self.line()
            depth = max(len(self.lists) - 1, 0)
            self.buf += "  " * depth

            if self.lists and self.lists[-1][0] == "ol":
                self.lists[-1][1] += 1
                self.buf += f"{self.lists[-1][1]}. "
            else:
                self.buf += "- "

        elif tag == "hr":
            self.block()
            self.buf += "---"
            self.block()

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3", "p", "blockquote"):
            self.block()

        elif tag == "div":
            self.line()

        elif tag in ("strong", "b"):
            self.buf += "**"

        elif tag in ("em", "i"):
            self.buf += "*"

        elif tag in ("s", "strike"):
            self.buf += "~~"

        elif tag == "code" and not self.in_pre:
            self.buf += "`"

        elif tag == "pre":
            self.line()
            self.buf += "```"
            self.block()
            self.in_pre = False

        elif tag == "a":
            if self.href:
                self.buf += f"]({self.href})"
                self.href = None

        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            if self.lists:
                self.line()
            else:
                self.block()

        elif tag == "li":
            self.line()

    def handle_data(self, data):
        if self.in_pre:
            self.buf += data
        else:
            self.buf += data.replace("\n", " ")

    def markdown(self):
        text = re.sub(r"\n{3,}", "\n\n", self.buf)
        return text.strip() + "\n"


def html_to_md(html):
    writer = MarkdownWriter()
    writer.feed(html or "")
    writer.close()
    return writer.markdown()

# ======================================================================
# SMALL VALIDATORS
# ======================================================================

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def valid_date(value):
    if not isinstance(value, str) or not DATE_RE.match(value):
        return False

    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def valid_time(value):
    if not isinstance(value, str) or not TIME_RE.match(value):
        return False

    hours, minutes = int(value[:2]), int(value[3:])
    return 0 <= hours <= 23 and 0 <= minutes <= 59


def safe_filename(title, extension):
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", title or "").strip(" ._")
    return (name[:80] or "note") + extension


def folder_exists(db, folder_id):
    if folder_id is None:
        return False

    return db.execute(
        "SELECT 1 FROM folders WHERE id = ?", (folder_id,)
    ).fetchone() is not None


# ======================================================================
# UPLOADED FILES (images and PDFs)
# ======================================================================

def detect_upload_type(raw):
    """Work out what a file really is from its first bytes (not its name)."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"

    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"

    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"

    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp", "image/webp"

    if raw.startswith(b"%PDF-"):
        return ".pdf", "application/pdf"

    return None


def write_upload(name, raw):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with open(os.path.join(UPLOAD_DIR, name), "wb") as handle:
        handle.write(raw)


def save_upload(raw, extension):
    name = secrets.token_hex(16) + extension
    write_upload(name, raw)
    return name


def cleanup_uploads():
    """Delete stored files that no page uses any more."""
    if not os.path.isdir(UPLOAD_DIR):
        return

    rows = get_db().execute("SELECT content FROM notes").fetchall()
    used = " ".join(row["content"] for row in rows)

    # Files younger than an hour are kept: they may belong to a page
    # that hasn't finished autosaving yet.
    cutoff = time.time() - 3600

    for name in os.listdir(UPLOAD_DIR):
        if not UPLOAD_NAME_RE.match(name) or name in used:
            continue

        path = os.path.join(UPLOAD_DIR, name)

        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def inline_images(html):
    """Make exported HTML self-contained: embed pictures, drop file links."""

    def embed(match):
        name = match.group(1)

        try:
            with open(os.path.join(UPLOAD_DIR, name), "rb") as handle:
                data = base64.b64encode(handle.read()).decode("ascii")
        except OSError:
            return match.group(0)

        mime = MIME_BY_EXT[os.path.splitext(name)[1]]
        return f'src="data:{mime};base64,{data}"'

    html = re.sub(r'src="/uploads/([0-9a-f]{32}\.(?:png|jpg|gif|webp))"', embed, html)
    html = re.sub(r'<a href="/uploads/[0-9a-f]{32}\.pdf"[^>]*>(.*?)</a>', r"\1", html)

    return html


# ======================================================================
# MAIN PAGE
# ======================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/upload-image")
def upload_image():
    upload = request.files.get("file")

    if upload is None:
        return jsonify({"error": "Choose an image."}), 400

    raw = upload.read()
    kind = detect_upload_type(raw)

    if kind is None or kind[0] == ".pdf":
        return jsonify({
            "error": "That file isn't a PNG, JPEG, GIF or WebP image."
        }), 400

    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify({"error": "That image is over 10 MB."}), 413

    name = save_upload(raw, kind[0])

    return jsonify({"url": f"/uploads/{name}"}), 201


@app.get("/uploads/<name>")
def serve_upload(name):
    if not UPLOAD_NAME_RE.match(name):
        abort(404)

    if not os.path.isfile(os.path.join(UPLOAD_DIR, name)):
        abort(404)

    response = send_from_directory(UPLOAD_DIR, name)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "private, max-age=86400"

    if name.endswith(".pdf"):
        # Download PDFs instead of opening them inside the app's origin.
        response.headers["Content-Disposition"] = 'attachment; filename="document.pdf"'

    return response


# ======================================================================
# FOLDERS (CLASSES)
# ======================================================================

@app.get("/api/folders")
def list_folders():
    rows = get_db().execute(
        """
        SELECT f.id, f.name, COUNT(n.id) AS note_count
        FROM folders f
        LEFT JOIN notes n ON n.folder_id = f.id
        GROUP BY f.id
        ORDER BY f.name COLLATE NOCASE
        """
    ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.post("/api/folders")
def create_folder():
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()[:60]

    if not name:
        return jsonify({"error": "Please enter a class name."}), 400

    db = get_db()

    cursor = db.execute("INSERT INTO folders (name) VALUES (?)", (name,))
    db.commit()

    return jsonify({"id": cursor.lastrowid, "name": name}), 201


@app.put("/api/folders/<int:folder_id>")
def rename_folder(folder_id):
    data = request.get_json(silent=True) or {}

    name = str(data.get("name", "")).strip()[:60]

    if not name:
        return jsonify({"error": "Please enter a class name."}), 400

    db = get_db()

    cursor = db.execute(
        "UPDATE folders SET name = ? WHERE id = ?", (name, folder_id)
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Class not found"}), 404

    return jsonify({"success": True})


@app.delete("/api/folders/<int:folder_id>")
def delete_folder(folder_id):
    db = get_db()

    # Pages are kept; they just become unfiled.
    db.execute(
        "UPDATE notes SET folder_id = NULL WHERE folder_id = ?", (folder_id,)
    )

    # A deleted class can no longer be shared.
    db.execute("DELETE FROM shares WHERE folder_id = ?", (folder_id,))

    cursor = db.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Class not found"}), 404

    return jsonify({"success": True})


# ======================================================================
# NOTES
# ======================================================================

def insert_note(db, title, html, folder_id=None, updated_at=None):
    cursor = db.execute(
        """
        INSERT INTO notes (title, content, folder_id, updated_at, is_html)
        VALUES (?, ?, ?, ?, 1)
        """,
        (
            (title or "Untitled").strip()[:200] or "Untitled",
            html,
            folder_id,
            updated_at or now_string(),
        ),
    )
    return cursor.lastrowid


@app.get("/api/notes")
def list_notes():
    rows = get_db().execute(
        """
        SELECT id, title, updated_at, folder_id, content, is_html
        FROM notes
        ORDER BY updated_at DESC, id DESC
        """
    ).fetchall()

    notes = []

    for row in rows:
        note = {
            "id": row["id"],
            "title": row["title"],
            "updated_at": row["updated_at"],
            "folder_id": row["folder_id"],
            "preview": " ".join(content_text(row).split())[:120],
        }
        notes.append(note)

    return jsonify(notes)


@app.post("/api/notes")
def create_note():
    data = request.get_json(silent=True) or {}

    db = get_db()

    folder_id = data.get("folder_id")

    if not folder_exists(db, folder_id):
        folder_id = None

    note_id = insert_note(db, "Untitled", "", folder_id)
    db.commit()

    return jsonify({"id": note_id}), 201


@app.get("/api/notes/<int:note_id>")
def get_note(note_id):
    row = get_db().execute(
        "SELECT * FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if row is None:
        return jsonify({"error": "Note not found"}), 404

    note = dict(row)
    note["content"] = content_html(row)   # always safe HTML
    note.pop("is_html", None)

    return jsonify(note)


@app.put("/api/notes/<int:note_id>")
def update_note(note_id):
    data = request.get_json(silent=True) or {}

    title = str(data.get("title", "Untitled")).strip()[:200] or "Untitled"
    content = clean_html(str(data.get("content", "")))

    db = get_db()

    cursor = db.execute(
        """
        UPDATE notes
        SET title = ?, content = ?, is_html = 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (title, content, note_id),
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"success": True})


@app.put("/api/notes/<int:note_id>/move")
def move_note(note_id):
    data = request.get_json(silent=True) or {}

    folder_id = data.get("folder_id")   # None means "Unfiled"

    db = get_db()

    if folder_id is not None and not folder_exists(db, folder_id):
        return jsonify({"error": "Class not found"}), 404

    cursor = db.execute(
        "UPDATE notes SET folder_id = ? WHERE id = ?", (folder_id, note_id)
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"success": True})


@app.delete("/api/notes/<int:note_id>")
def delete_note(note_id):
    db = get_db()

    db.execute("DELETE FROM comments WHERE note_id = ?", (note_id,))
    db.execute("DELETE FROM shares WHERE note_id = ?", (note_id,))

    cursor = db.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Note not found"}), 404

    cleanup_uploads()

    return jsonify({"success": True})

# ======================================================================
# COMMENTS
# ======================================================================

@app.get("/api/notes/<int:note_id>/comments")
def list_comments(note_id):
    rows = get_db().execute(
        """
        SELECT id, author, body, source, created_at
        FROM comments
        WHERE note_id = ?
        ORDER BY id
        """,
        (note_id,),
    ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.post("/api/notes/<int:note_id>/comments")
def add_comment(note_id):
    data = request.get_json(silent=True) or {}

    body = str(data.get("body", "")).strip()[:1000]

    if not body:
        return jsonify({"error": "Please write a comment."}), 400

    db = get_db()

    exists = db.execute(
        "SELECT 1 FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if exists is None:
        return jsonify({"error": "Note not found"}), 404

    cursor = db.execute(
        "INSERT INTO comments (note_id, author, body, source) VALUES (?, 'You', ?, 'owner')",
        (note_id, body),
    )
    db.commit()

    return jsonify({"id": cursor.lastrowid}), 201


@app.delete("/api/comments/<int:comment_id>")
def delete_comment(comment_id):
    db = get_db()

    cursor = db.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Comment not found"}), 404

    return jsonify({"success": True})


# ======================================================================
# TO-DO LIST
# ======================================================================

@app.get("/api/todos")
def list_todos():
    rows = get_db().execute(
        """
        SELECT id, text, done, due_date, created_at
        FROM todos
        ORDER BY done, due_date IS NULL, due_date, id
        """
    ).fetchall()

    todos = [dict(row) for row in rows]

    for todo in todos:
        todo["done"] = bool(todo["done"])

    return jsonify(todos)


@app.post("/api/todos")
def create_todo():
    data = request.get_json(silent=True) or {}

    text = str(data.get("text", "")).strip()[:300]
    due_date = data.get("due_date") or None

    if not text:
        return jsonify({"error": "Please enter a task."}), 400

    if due_date is not None and not valid_date(due_date):
        return jsonify({"error": "That date isn't valid."}), 400

    db = get_db()

    cursor = db.execute(
        "INSERT INTO todos (text, due_date) VALUES (?, ?)", (text, due_date)
    )
    db.commit()

    return jsonify({"id": cursor.lastrowid}), 201


@app.put("/api/todos/<int:todo_id>")
def update_todo(todo_id):
    data = request.get_json(silent=True) or {}

    fields = []
    values = []

    if "done" in data:
        fields.append("done = ?")
        values.append(1 if data["done"] else 0)

    if "text" in data:
        text = str(data["text"]).strip()[:300]

        if not text:
            return jsonify({"error": "Please enter a task."}), 400

        fields.append("text = ?")
        values.append(text)

    if "due_date" in data:
        due_date = data["due_date"] or None

        if due_date is not None and not valid_date(due_date):
            return jsonify({"error": "That date isn't valid."}), 400

        fields.append("due_date = ?")
        values.append(due_date)

    if not fields:
        return jsonify({"error": "Nothing to update."}), 400

    values.append(todo_id)

    db = get_db()

    cursor = db.execute(
        f"UPDATE todos SET {', '.join(fields)} WHERE id = ?", values
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({"success": True})


@app.delete("/api/todos/<int:todo_id>")
def delete_todo(todo_id):
    db = get_db()

    cursor = db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Task not found"}), 404

    return jsonify({"success": True})


@app.post("/api/todos/clear-completed")
def clear_completed_todos():
    db = get_db()

    cursor = db.execute("DELETE FROM todos WHERE done = 1")
    db.commit()

    return jsonify({"deleted": cursor.rowcount})


# ======================================================================
# CALENDAR EVENTS
# ======================================================================

@app.get("/api/events")
def list_events():
    rows = get_db().execute(
        "SELECT id, title, date, time FROM events ORDER BY date, time, id"
    ).fetchall()

    return jsonify([dict(row) for row in rows])


@app.post("/api/events")
def create_event():
    data = request.get_json(silent=True) or {}

    title = str(data.get("title", "")).strip()[:120]
    date = data.get("date")
    time_value = data.get("time") or ""

    if not title:
        return jsonify({"error": "Please enter an event title."}), 400

    if not valid_date(date):
        return jsonify({"error": "Please choose a valid date."}), 400

    if time_value and not valid_time(time_value):
        return jsonify({"error": "That time isn't valid."}), 400

    db = get_db()

    cursor = db.execute(
        "INSERT INTO events (title, date, time) VALUES (?, ?, ?)",
        (title, date, time_value),
    )
    db.commit()

    return jsonify({"id": cursor.lastrowid}), 201


@app.delete("/api/events/<int:event_id>")
def delete_event(event_id):
    db = get_db()

    cursor = db.execute("DELETE FROM events WHERE id = ?", (event_id,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Event not found"}), 404

    return jsonify({"success": True})


# ======================================================================
# SHARING
#
# A share is a secret link. Anyone who has the link can view the page
# (or class) in read-only mode, and optionally leave comments.
# ======================================================================

def share_to_dict(row):
    if row is None:
        return None

    share = dict(row)
    share["allow_comments"] = bool(share["allow_comments"])
    return share


@app.get("/api/shares")
def get_share_for_target():
    note_id = request.args.get("note_id", type=int)
    folder_id = request.args.get("folder_id", type=int)

    db = get_db()

    if note_id is not None:
        row = db.execute(
            "SELECT * FROM shares WHERE note_id = ?", (note_id,)
        ).fetchone()
    elif folder_id is not None:
        row = db.execute(
            "SELECT * FROM shares WHERE folder_id = ?", (folder_id,)
        ).fetchone()
    else:
        return jsonify({"error": "Give a note_id or folder_id."}), 400

    return jsonify({"share": share_to_dict(row)})


@app.post("/api/shares")
def create_share():
    data = request.get_json(silent=True) or {}

    note_id = data.get("note_id")
    folder_id = data.get("folder_id")
    allow_comments = 1 if data.get("allow_comments") else 0

    db = get_db()

    if note_id is not None:
        exists = db.execute(
            "SELECT 1 FROM notes WHERE id = ?", (note_id,)
        ).fetchone()

        if exists is None:
            return jsonify({"error": "Note not found"}), 404

        existing = db.execute(
            "SELECT * FROM shares WHERE note_id = ?", (note_id,)
        ).fetchone()

    elif folder_id is not None:
        if not folder_exists(db, folder_id):
            return jsonify({"error": "Class not found"}), 404

        existing = db.execute(
            "SELECT * FROM shares WHERE folder_id = ?", (folder_id,)
        ).fetchone()

    else:
        return jsonify({"error": "Give a note_id or folder_id."}), 400

    if existing is not None:
        return jsonify({"share": share_to_dict(existing)})

    token = secrets.token_urlsafe(16)

    db.execute(
        """
        INSERT INTO shares (token, note_id, folder_id, allow_comments)
        VALUES (?, ?, ?, ?)
        """,
        (token, note_id, folder_id, allow_comments),
    )
    db.commit()

    row = db.execute(
        "SELECT * FROM shares WHERE token = ?", (token,)
    ).fetchone()

    return jsonify({"share": share_to_dict(row)}), 201


@app.put("/api/shares/<token>")
def update_share(token):
    data = request.get_json(silent=True) or {}

    db = get_db()

    cursor = db.execute(
        "UPDATE shares SET allow_comments = ? WHERE token = ?",
        (1 if data.get("allow_comments") else 0, token),
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Share not found"}), 404

    return jsonify({"success": True})


@app.delete("/api/shares/<token>")
def delete_share(token):
    db = get_db()

    cursor = db.execute("DELETE FROM shares WHERE token = ?", (token,))
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Share not found"}), 404

    return jsonify({"success": True})


# ---------- Public (read-only) pages ----------

def find_share(token):
    return get_db().execute(
        "SELECT * FROM shares WHERE token = ?", (token,)
    ).fetchone()


def note_in_share(share, note_id):
    """Is this note covered by this share link?"""
    if share["note_id"] is not None:
        return share["note_id"] == note_id

    row = get_db().execute(
        "SELECT 1 FROM notes WHERE id = ? AND folder_id = ?",
        (note_id, share["folder_id"]),
    ).fetchone()

    return row is not None


def render_shared_note(share, token, note_id):
    db = get_db()

    note = db.execute(
        "SELECT * FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if note is None:
        return render_template("share.html", mode="missing"), 404

    comments = db.execute(
        """
        SELECT author, body, source, created_at
        FROM comments
        WHERE note_id = ?
        ORDER BY id
        """,
        (note_id,),
    ).fetchall()

    back_url = None
    back_label = None

    if share["folder_id"] is not None:
        folder = db.execute(
            "SELECT name FROM folders WHERE id = ?", (share["folder_id"],)
        ).fetchone()

        back_url = f"/share/{token}"
        back_label = folder["name"] if folder else "Back"

    return render_template(
        "share.html",
        mode="note",
        title=note["title"],
        content=content_html(note),
        token=token,
        note_id=note_id,
        allow_comments=bool(share["allow_comments"]),
        comments=[dict(c) for c in comments],
        back_url=back_url,
        back_label=back_label,
    )


@app.get("/share/<token>")
def view_share(token):
    share = find_share(token)

    if share is None:
        return render_template("share.html", mode="missing"), 404

    if share["note_id"] is not None:
        return render_shared_note(share, token, share["note_id"])

    db = get_db()

    folder = db.execute(
        "SELECT name FROM folders WHERE id = ?", (share["folder_id"],)
    ).fetchone()

    if folder is None:
        return render_template("share.html", mode="missing"), 404

    rows = db.execute(
        """
        SELECT id, title, content, is_html
        FROM notes
        WHERE folder_id = ?
        ORDER BY title COLLATE NOCASE
        """,
        (share["folder_id"],),
    ).fetchall()

    notes = [
        {
            "id": row["id"],
            "title": row["title"],
            "preview": " ".join(content_text(row).split())[:140],
        }
        for row in rows
    ]

    return render_template(
        "share.html",
        mode="folder",
        title=folder["name"],
        notes=notes,
        token=token,
    )


@app.get("/share/<token>/<int:note_id>")
def view_shared_note(token, note_id):
    share = find_share(token)

    if share is None or not note_in_share(share, note_id):
        return render_template("share.html", mode="missing"), 404

    return render_shared_note(share, token, note_id)


@app.post("/share/<token>/<int:note_id>/comments")
def add_shared_comment(token, note_id):
    share = find_share(token)

    if share is None or not note_in_share(share, note_id):
        return jsonify({"error": "This link is no longer available."}), 404

    if not share["allow_comments"]:
        return jsonify({"error": "Comments are turned off for this link."}), 403

    data = request.get_json(silent=True) or {}

    author = str(data.get("author", "")).strip()[:40] or "Guest"
    body = str(data.get("body", "")).strip()[:1000]

    if not body:
        return jsonify({"error": "Please write a comment."}), 400

    db = get_db()

    db.execute(
        """
        INSERT INTO comments (note_id, author, body, source)
        VALUES (?, ?, ?, 'shared')
        """,
        (note_id, author, body),
    )
    db.commit()

    return jsonify({"success": True}), 201

# ======================================================================
# EXPORT
# ======================================================================

EXPORT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: Arial, Helvetica, sans-serif; max-width: 720px;
         margin: 40px auto; padding: 0 20px; color: #37352f; line-height: 1.7; }}
  h1 {{ font-size: 2em; }}
  blockquote {{ border-left: 3px solid #ddd; margin-left: 0; padding-left: 14px; color: #6b6b67; }}
  pre {{ background: #f5f5f3; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  code {{ background: #f5f5f3; padding: 1px 4px; border-radius: 3px; }}
  pre code {{ padding: 0; background: none; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>
"""


@app.get("/api/notes/<int:note_id>/export")
def export_note(note_id):
    row = get_db().execute(
        "SELECT * FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if row is None:
        return jsonify({"error": "Note not found"}), 404

    fmt = request.args.get("format", "html").lower()

    html = content_html(row)
    title = row["title"] or "Untitled"

    if fmt == "md":
        body = f"# {title}\n\n" + html_to_md(html)
        mimetype, extension = "text/markdown", ".md"

    elif fmt == "txt":
        body = f"{title}\n\n" + html_to_text(html) + "\n"
        mimetype, extension = "text/plain", ".txt"

    elif fmt == "html":
        body = EXPORT_PAGE.format(title=escape(title), body=inline_images(html))
        mimetype, extension = "text/html", ".html"

    else:
        return jsonify({"error": "Unknown format."}), 400

    filename = safe_filename(title, extension)

    return Response(
        body,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@app.get("/api/export/all")
def export_all():
    db = get_db()

    payload = {
        "app": "Notes AI",
        "version": 1,
        "exported_at": now_string(),
        "folders": [
            dict(r) for r in db.execute("SELECT id, name FROM folders")
        ],
        "notes": [
            {
                "id": r["id"],
                "title": r["title"],
                "content": content_html(r),
                "folder_id": r["folder_id"],
                "updated_at": r["updated_at"],
            }
            for r in db.execute(
                "SELECT id, title, content, is_html, folder_id, updated_at FROM notes"
            )
        ],
        "comments": [
            dict(r) for r in db.execute(
                "SELECT note_id, author, body, source, created_at FROM comments"
            )
        ],
        "todos": [
            {**dict(r), "done": bool(r["done"])}
            for r in db.execute("SELECT text, done, due_date FROM todos")
        ],
        "events": [
            dict(r) for r in db.execute("SELECT title, date, time FROM events")
        ],
    }

    # Include every picture and PDF the pages use, so a restore is complete.
    used = set()

    for note in payload["notes"]:
        used.update(UPLOAD_FIND_RE.findall(note["content"]))

    files = {}

    for name in sorted(used):
        try:
            with open(os.path.join(UPLOAD_DIR, name), "rb") as handle:
                files[name] = base64.b64encode(handle.read()).decode("ascii")
        except OSError:
            pass

    payload["files"] = files

    filename = f"notes-ai-backup-{datetime.now().strftime('%Y-%m-%d')}.json"

    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ======================================================================
# IMPORT
# ======================================================================

def as_list(value):
    return value if isinstance(value, list) else []


def restore_backup(db, data):
    """Add everything from a backup file. Nothing existing is changed."""
    if not isinstance(data, dict):
        raise ValueError("Not a backup file.")

    counts = {
        "folders": 0, "notes": 0, "comments": 0,
        "todos": 0, "events": 0, "files": 0,
    }

    # Pictures and PDFs first, so restored pages can show them.
    files = data.get("files")

    for name, encoded in (files.items() if isinstance(files, dict) else []):
        if not isinstance(name, str) or not UPLOAD_NAME_RE.match(name):
            continue

        if not isinstance(encoded, str) or os.path.exists(os.path.join(UPLOAD_DIR, name)):
            continue

        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception:
            continue

        kind = detect_upload_type(raw)

        if kind is None or kind[0] != os.path.splitext(name)[1]:
            continue

        write_upload(name, raw)
        counts["files"] += 1

    # Classes: reuse one with the same name, otherwise create it.
    folder_map = {}

    for item in as_list(data.get("folders")):
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()[:60]

        if not name:
            continue

        existing = db.execute(
            "SELECT id FROM folders WHERE name = ? COLLATE NOCASE", (name,)
        ).fetchone()

        if existing:
            new_id = existing["id"]
        else:
            new_id = db.execute(
                "INSERT INTO folders (name) VALUES (?)", (name,)
            ).lastrowid
            counts["folders"] += 1

        folder_map[item.get("id")] = new_id

    # Notes
    note_map = {}

    for item in as_list(data.get("notes")):
        if not isinstance(item, dict):
            continue

        stamp = item.get("updated_at")

        new_id = insert_note(
            db,
            str(item.get("title", "Untitled")),
            clean_html(str(item.get("content", ""))),
            folder_map.get(item.get("folder_id")),
            stamp if isinstance(stamp, str) and STAMP_RE.match(stamp) else None,
        )

        note_map[item.get("id")] = new_id
        counts["notes"] += 1

    # Comments
    for item in as_list(data.get("comments")):
        if not isinstance(item, dict):
            continue

        new_note = note_map.get(item.get("note_id"))
        body = str(item.get("body", "")).strip()[:1000]

        if new_note is None or not body:
            continue

        source = "shared" if item.get("source") == "shared" else "owner"

        db.execute(
            "INSERT INTO comments (note_id, author, body, source) VALUES (?, ?, ?, ?)",
            (new_note, str(item.get("author", "You"))[:40] or "You", body, source),
        )
        counts["comments"] += 1

    # Tasks
    for item in as_list(data.get("todos")):
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()[:300]

        if not text:
            continue

        due = item.get("due_date")

        db.execute(
            "INSERT INTO todos (text, done, due_date) VALUES (?, ?, ?)",
            (text, 1 if item.get("done") else 0, due if valid_date(due) else None),
        )
        counts["todos"] += 1

    # Events
    for item in as_list(data.get("events")):
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()[:120]
        date = item.get("date")
        time_value = item.get("time") or ""

        if not title or not valid_date(date):
            continue

        db.execute(
            "INSERT INTO events (title, date, time) VALUES (?, ?, ?)",
            (title, date, time_value if valid_time(time_value) else ""),
        )
        counts["events"] += 1

    return counts

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

LIST_START_RE = re.compile(r"^([\u2022\u25cf\u25aa*\-]|\d+[.)])\s")

TRANSCRIBE_PROMPT = """Transcribe all of the text in this {what}.

- Write it as Markdown: use # headings, - bullet lists and numbered lists
  wherever the original has them.
- Keep the original wording and order. Do not summarize or add commentary.
- For handwriting, do your best to read it.
- If there is no readable text at all, reply with exactly: NO TEXT"""


def transcribe_with_ai(raw, mime_type, what):
    """Ask Gemini to read the text in an image or PDF.

    Returns (html, error). html is "" when the file has no readable text.
    """
    contents = [
        types.Part.from_bytes(data=raw, mime_type=mime_type),
        TRANSCRIBE_PROMPT.format(what=what),
    ]

    text, error = generate_ai_response(contents)

    if error:
        return None, error

    text = text.strip()

    # Models sometimes wrap the whole answer in a code fence.
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    if text.upper().startswith("NO TEXT"):
        return "", None

    return clean_html(md_to_html(text)), None


def pdf_text_pages(raw):
    """Text of each PDF page, or None if it can't be read directly."""
    if PdfReader is None:
        return None

    reader = PdfReader(io.BytesIO(raw))

    if reader.is_encrypted and reader.decrypt("") == 0:
        return None

    pages = []

    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    return pages


def reflow_pdf_text(text):
    """PDFs break lines wherever the layout does. Join them back into paragraphs."""
    paragraphs = []
    current = []

    def flush():
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    for line in text.replace("\r", "").split("\n"):
        line = line.strip()

        if not line:
            flush()
            continue

        if current:
            previous = current[-1]

            if (
                LIST_START_RE.match(line)
                or previous.endswith((".", "!", "?", ":", ";"))
                or len(previous) < 40
            ):
                flush()

        current.append(line)

    flush()

    return paragraphs


def pdf_pages_to_html(pages, limit=400_000):
    parts = []
    total = 0

    for number, text in enumerate(pages):
        paragraphs = reflow_pdf_text(text)

        if not paragraphs:
            continue

        if parts:
            parts.append("<hr>")

        for paragraph in paragraphs:
            total += len(paragraph)

            if total > limit:
                parts.append("<p><i>(The rest of this PDF was left out because it is very long.)</i></p>")
                return "".join(parts)

            parts.append("<p>" + escape(paragraph) + "</p>")

    return "".join(parts)


def import_pdf(db, raw, filename, stem, folder_id, warnings):
    if not raw.startswith(b"%PDF-"):
        raise ValueError("Not a PDF file.")

    if len(raw) > MAX_PDF_BYTES:
        raise ValueError("PDF is over 30 MB.")

    stored = save_upload(raw, ".pdf")
    body = ""

    try:
        pages = pdf_text_pages(raw)
    except Exception as error:
        print("PDF read error:", filename, error)
        pages = None

    has_text = (
        pages is not None
        and sum(len(page.strip()) for page in pages) >= 40 * max(len(pages), 1)
    )

    if has_text:
        body = pdf_pages_to_html(pages)

    elif len(raw) > MAX_AI_FILE_BYTES:
        warnings.append(
            f"{filename}: it looks scanned and is too large for AI reading "
            "(limit about 18 MB), so only the original file was attached."
        )

    else:
        # Scanned PDF (or pypdf isn't installed): let Gemini read it.
        html, error = transcribe_with_ai(raw, "application/pdf", "document")

        if error:
            warnings.append(
                f"{filename}: couldn't read the text ({error}) "
                "The original PDF is attached."
            )
        elif not html:
            warnings.append(f"{filename}: no readable text was found.")
        else:
            body = html

    link = (
        f'<p><a href="/uploads/{stored}">'
        f"\U0001F4CE Original PDF: {escape(filename)}</a></p>"
    )

    insert_note(db, stem, clean_html(link + body), folder_id)


def import_image(db, raw, filename, stem, folder_id, warnings):
    kind = detect_upload_type(raw)

    if kind is None or kind[0] == ".pdf":
        raise ValueError("Not a supported image.")

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is over 10 MB.")

    extension, mime_type = kind
    stored = save_upload(raw, extension)

    html = f'<p><img src="/uploads/{stored}" alt="{escape(stem, quote=True)}"></p>'

    text_html, error = transcribe_with_ai(raw, mime_type, "image")

    if error:
        warnings.append(
            f"{filename}: added the image, but couldn't read its text ({error})"
        )
    elif text_html:
        html += text_html

    insert_note(db, stem, clean_html(html), folder_id)


@app.post("/api/import")
def import_files():
    files = request.files.getlist("files")

    if not files:
        return jsonify({"error": "Choose at least one file."}), 400

    db = get_db()

    folder_id = request.form.get("folder_id", type=int)

    if not folder_exists(db, folder_id):
        folder_id = None

    created = 0
    skipped = []
    warnings = []
    restored = None

    for upload in files:
        filename = os.path.basename(upload.filename or "Untitled")
        stem, extension = os.path.splitext(filename)
        extension = extension.lower()
        stem = stem or "Untitled"

        try:
            raw = upload.read()

            if extension == ".pdf":
                import_pdf(db, raw, filename, stem, folder_id, warnings)
                created += 1

            elif extension in IMAGE_EXTENSIONS:
                import_image(db, raw, filename, stem, folder_id, warnings)
                created += 1

            elif extension in (".txt", ".md", ".markdown", ".html", ".htm", ".json"):
                text = raw.decode("utf-8-sig", errors="replace")

                if extension == ".txt":
                    insert_note(db, stem, text_to_html(text), folder_id)
                    created += 1

                elif extension in (".md", ".markdown"):
                    insert_note(db, stem, clean_html(md_to_html(text)), folder_id)
                    created += 1

                elif extension in (".html", ".htm"):
                    match = re.search(
                        r"<title[^>]*>(.*?)</title>", text, re.IGNORECASE | re.DOTALL
                    )
                    title = unescape(match.group(1)).strip() if match else stem

                    insert_note(db, title or stem, clean_html(text), folder_id)
                    created += 1

                else:
                    counts = restore_backup(db, json.loads(text))

                    if restored is None:
                        restored = counts
                    else:
                        for key in counts:
                            restored[key] += counts[key]

            else:
                skipped.append(filename)

        except Exception as error:
            print("Import error:", filename, error)
            skipped.append(filename)

        # Save after every file, so a slow AI call on the next file
        # never keeps the database locked.
        db.commit()

    return jsonify({
        "notes_created": created,
        "restored": restored,
        "skipped": skipped,
        "warnings": warnings,
    })

# ======================================================================
# AI HELPER (retry + model fallback)
# ======================================================================

def generate_ai_response(prompt):
    last_error = ""

    for model in MODELS:

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                if response.text:
                    return response.text, None

                last_error = "empty response"
                break  # try the next model

            except Exception as error:
                last_error = str(error)
                print(
                    f"Gemini error ({model}, attempt {attempt + 1}):",
                    error
                )

                # A bad API key won't be fixed by another model.
                if "401" in last_error or "403" in last_error:
                    return None, "There is a problem with the Gemini API key."

                # Overloaded: wait briefly, retry the same model once.
                if "503" in last_error or "UNAVAILABLE" in last_error:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                # Anything else (429, 404, ...): next model.
                break

    if "503" in last_error or "UNAVAILABLE" in last_error:
        return None, "AI is currently busy. Please try again in a moment."

    if "429" in last_error:
        return None, (
            "The AI request limit has been reached. "
            "Please wait a moment and try again."
        )

    return None, "Something went wrong while contacting the AI."


def load_note(note_id):
    """Load a note with its content converted to plain text for the AI."""
    row = get_db().execute(
        "SELECT * FROM notes WHERE id = ?", (note_id,)
    ).fetchone()

    if row is None:
        return None

    note = dict(row)
    note["content"] = content_text(row)

    return note


# ======================================================================
# SPLIT LARGE NOTES
# ======================================================================

def split_text(text, max_chars=12000):
    chunks = []

    text = text.strip()

    while len(text) > max_chars:

        split_point = text.rfind("\n", 0, max_chars)

        if split_point == -1:
            split_point = text.rfind(" ", 0, max_chars)

        if split_point == -1:
            split_point = max_chars

        chunk = text[:split_point].strip()

        if chunk:
            chunks.append(chunk)

        text = text[split_point:].strip()

    if text:
        chunks.append(text)

    return chunks


# ======================================================================
# SUMMARIZE NOTE
# ======================================================================

@app.post("/api/notes/<int:note_id>/summarize")
def summarize_note(note_id):

    note = load_note(note_id)

    if note is None:
        return jsonify({"error": "Note not found"}), 404

    content = note["content"].strip()

    if not content:
        return jsonify({"error": "The note is empty."}), 400

    chunks = split_text(content)

    summaries = []

    for chunk_number, chunk in enumerate(chunks, start=1):

        prompt = f"""
You are an AI study assistant.

Summarize the following section of a student's notes.

Focus on the most important concepts and facts.

Only use information contained in the notes.
Do not add information that is not present.

This is section {chunk_number} of {len(chunks)}.

Notes:

{chunk}
"""

        result, error = generate_ai_response(prompt)

        if error:
            return jsonify({"error": error}), 503

        summaries.append(result)

    if len(summaries) == 1:
        return jsonify({"result": summaries[0]})

    combined_summaries = "\n\n".join(summaries)

    final_prompt = f"""
You are an AI study assistant.

Below are summaries from different sections
of one large set of notes.

Combine these section summaries into one
clear final summary.

Requirements:

- Include the most important information.
- Use clear bullet points.
- Remove repeated information.
- Keep related ideas together.
- Do not add information that was not
  included in the section summaries.

Section summaries:

{combined_summaries}
"""

    final_result, error = generate_ai_response(final_prompt)

    if error:
        return jsonify({"error": error}), 503

    return jsonify({"result": final_result})


# ======================================================================
# STUDY QUESTIONS
# ======================================================================

@app.post("/api/notes/<int:note_id>/questions")
def generate_questions(note_id):

    note = load_note(note_id)

    if note is None:
        return jsonify({"error": "Note not found"}), 404

    content = note["content"].strip()

    if not content:
        return jsonify({"error": "The note is empty."}), 400

    chunks = split_text(content)

    if len(chunks) > 1:

        section_summaries = []

        for chunk in chunks:

            summary_prompt = f"""
Summarize this section of study notes.

Keep the important facts, concepts,
definitions, and relationships.

Only use information from the notes.

Notes:

{chunk}
"""

            result, error = generate_ai_response(summary_prompt)

            if error:
                return jsonify({"error": error}), 503

            section_summaries.append(result)

        study_content = "\n\n".join(section_summaries)

    else:
        study_content = content

    prompt = f"""
You are an AI study assistant.

Using the following notes, create
5 useful study questions.

Include a mixture of:

- recall questions
- understanding questions
- application questions

Number the questions 1 through 5.

Do not provide the answers.

Only create questions based on
information contained in the notes.

Title:
{note["title"]}

Notes:
{study_content}
"""

    result, error = generate_ai_response(prompt)

    if error:
        return jsonify({"error": error}), 503

    return jsonify({"result": result})


# ======================================================================
# ASK AI ABOUT NOTE
# ======================================================================

def build_rules(allow_outside):
    if allow_outside:
        return """The student has allowed you to use your own general
knowledge in addition to their notes.

- First, say what the notes contain about the question under the
  heading "From your notes:". If the notes say nothing relevant,
  write "Your notes don't cover this."
- Then add any extra information under the heading "Beyond your notes:".
- Never present outside information as if it came from the notes."""

    return """Answer using ONLY the provided notes.

If the answer cannot be determined from the notes, say that the notes
do not contain enough information, and suggest turning on
"Allow outside knowledge" if the student wants more.

Do not use outside knowledge. Do not make up information."""


@app.post("/api/notes/<int:note_id>/ask")
def ask_about_note(note_id):

    note = load_note(note_id)

    if note is None:
        return jsonify({"error": "Note not found"}), 404

    data = request.get_json(silent=True) or {}

    question = str(data.get("question", "")).strip()
    allow_outside = bool(data.get("allow_outside", False))

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    content = note["content"].strip()

    if not content and not allow_outside:
        return jsonify({"error": "The note is empty."}), 400

    chunks = split_text(content)

    # Small notes: send the whole note.
    if len(chunks) <= 1:
        notes_text = content or "(The note is empty.)"

    # Large notes: pull relevant information from each section first.
    else:
        relevant = []

        for chunk_number, chunk in enumerate(chunks, start=1):

            scan_prompt = f"""
You are analyzing one section of a student's notes.

If this section contains information that helps answer the
student's question, provide that information.

If it does not, respond exactly with:

NO RELEVANT INFORMATION

Section {chunk_number}:

{chunk}

Student question:

{question}
"""

            result, error = generate_ai_response(scan_prompt)

            if error:
                return jsonify({"error": error}), 503

            if result.strip().upper() != "NO RELEVANT INFORMATION":
                relevant.append(result)

        if not relevant and not allow_outside:
            return jsonify({
                "result": "The notes do not contain enough "
                          "information to answer that question."
            })

        notes_text = (
            "\n\n".join(relevant)
            or "(Nothing relevant was found in the notes.)"
        )

    prompt = f"""
You are an AI assistant helping a student with their notes.

{build_rules(allow_outside)}

Title:
{note["title"]}

Notes:
{notes_text}

Question:
{question}
"""

    result, error = generate_ai_response(prompt)

    if error:
        return jsonify({"error": error}), 503

    return jsonify({"result": result})


# ======================================================================
# START APPLICATION
# ======================================================================

# Make sure the tables exist however the app is started.
init_db()

if __name__ == "__main__":
    app.run(debug=True)