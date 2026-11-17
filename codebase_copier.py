"""
Gemini Codebase Copier
======================

Put this file in your project folder and run it. A window opens where you:

  1. pick the Gemini model and your Google AI plan,
  2. press the big "Copy part 1" button, paste into a NEW Gemini chat, send,
  3. wait for Gemini to reply "Part 1/N received", press the button again
     for part 2, and so on until the final part.

The whole project is split in order (docs and entry point first, then the
rest) into the fewest possible parts, sized as evenly as possible. Every
part fits the per-message limit, including the instructions around it, and:

  * starts with the line   ===== PART n OF N - START =====
  * ends with the line     ===== PART n OF N - END =====
  * tells Gemini to check that the END line arrived (so a cut-off paste is
    reported instead of silently losing code),
  * Part 1 also carries the project tree, middle parts only ask Gemini to
    confirm, and the final part asks Gemini to analyze the whole project and
    confirm it is ready to help you code.

Your choices (model, plan, custom limit) are remembered in
~/.gemini_codebase_copier.json.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

# ═══════════════════════════ SETTINGS YOU CAN EDIT ═══════════════════════════

# ── How these numbers were derived (research checked 17 Nov 2026) ─────────────
# 1. Google publishes NO per-message limit for the Gemini app. Its "Gemini Apps
#    limits" help page (support.google.com/gemini/answer/16275805) gives only the
#    context window ("memory") per PLAN, not per model: no plan 32k tokens,
#    AI Plus 128k, AI Pro 1M, AI Ultra 1M.
# 2. The per-message cap comes from the gemini.google.com chat box itself, not the
#    model, so it is the same for every mode in the picker. Independent tests
#    (fast.io Oct 2026, text-splitter.com Aug 2026, terabox, oreate) all land on
#    ~30,000 characters reliably accepted (fast.io: ~32,000). Above that the
#    message is rejected ("message too long") or the END IS SILENTLY CUT OFF.
#    Some accounts report a lower ceiling.
# 3. All picker models (3.5 Flash-Lite, 4 Hydrogen, 3.1 Pro, 4 Argon, 4 Carbon)
#    accept 1M+ tokens through the API, so the model is never the bottleneck.
# => 30,000 characters per message for every mode. Every part ends with an END
#    line that Gemini checks, so if your account has a lower ceiling you will be
#    told ("PART n WAS CUT OFF") and can lower the number in the window.

GEMINI_PLANS: dict[str, int] = {          # plan -> memory per chat (tokens)
    "No plan (free)": 32_000,
    "Google AI Plus": 128_000,
    "Google AI Pro (incl. Student)": 1_000_000,
    "Google AI Ultra": 1_000_000,
}

# Model picker modes: max characters per message, and which plans can pick it.
GEMINI_MODELS: dict[str, dict] = {
    "Quick answers - 3.5 Flash-Lite": {"chars": 30_000, "plans": None},
    "Everyday help - 4 Hydrogen": {"chars": 30_000, "plans": None},
    "Research - 3.1 Pro": {"chars": 30_000, "plans": ["Google AI Plus", "Google AI Pro (incl. Student)", "Google AI Ultra"]},
    "Code & analysis - 4 Argon": {"chars": 30_000, "plans": ["Google AI Pro (incl. Student)", "Google AI Ultra"]},
    "Most complex work - 4 Carbon": {"chars": 30_000, "plans": ["Google AI Ultra"]},
}
DEFAULT_MODEL = "Code & analysis - 4 Argon"
SAFE_MESSAGE_CHARS = 30_000
CHARS_PER_TOKEN = 3.5  # typical for source code

# Warn when the whole project would use more than this share of Gemini's memory
# (it also needs room for its answers and your later questions).
CONTEXT_WARN_SHARE = 0.75

EXCLUDED_DIRS = {
    ".venv", "venv", "env", ".git", "__pycache__", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "node_modules",
    "build", "dist", "data",
}
INCLUDED_EXTENSIONS = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ini"}
INCLUDED_EXACT_NAMES = {".gitignore", "requirements.txt"}

# Reading order: root files first, then these folders, unknown folders, tests last.
FOLDER_ORDER = {"": 0, "core": 1, "domain": 2, "engine": 3, "strategies": 4, "visualization": 5, "tests": 90}
ROOT_FILE_ORDER = {"readme.md": 0, "requirements.txt": 1, "pyproject.toml": 2, "main.py": 3}

MAX_FILE_BYTES = 500_000  # bigger files are skipped (probably data, not code)

# ═════════════════════════════════════════════════════════════════════════════

SELF_PATH = Path(__file__).resolve()
SETTINGS_PATH = Path.home() / ".gemini_codebase_copier.json"
SEP = "\n\n"

LANG_BY_SUFFIX = {
    ".py": "python", ".md": "markdown", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".txt": "text", ".cfg": "ini", ".ini": "ini",
}


def ulen(text: str) -> int:
    """Length the way browsers count it (UTF-16 units) - what chat limits measure."""
    return len(text.encode("utf-16-le")) // 2


def estimate_tokens(text: str) -> int:
    """Rough Gemini token count: ~3.5 characters per token for code, 1 per non-Latin character."""
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return int((len(text) - non_ascii) / CHARS_PER_TOKEN + non_ascii)


# ─────────────────────────────── Scanning ────────────────────────────────────

class SourceFile:
    __slots__ = ("rel", "content")

    def __init__(self, rel: str, content: str) -> None:
        self.rel = rel
        self.content = content


class ScanResult:
    # Plain class on purpose: this script is often saved as "copy.py", which
    # shadows Python's built-in `copy` module that `dataclasses` imports.
    def __init__(self, root: Path, files: list[SourceFile], tree: str, total_lines: int, skipped: list[str]) -> None:
        self.root = root
        self.files = files
        self.tree = tree
        self.total_lines = total_lines
        self.skipped = skipped


def _sort_key(rel: str) -> tuple:
    parts = rel.split("/")
    top = parts[0] if len(parts) > 1 else ""
    root_rank = ROOT_FILE_ORDER.get(parts[-1].lower(), 9) if top == "" else 0
    return (FOLDER_ORDER.get(top, 50), root_rank, rel.lower())


def _build_tree(root_name: str, rel_paths: list[str]) -> str:
    tree: dict = {}
    for rel in rel_paths:
        node = tree
        *dirs, name = rel.split("/")
        for d in dirs:
            node = node.setdefault(d + "/", {})
        node[name] = None

    lines = [root_name + "/"]

    def walk(node: dict, prefix: str) -> None:
        keys = sorted(node, key=lambda k: (not k.endswith("/"), k.lower()))
        for i, key in enumerate(keys):
            last = i == len(keys) - 1
            lines.append(prefix + ("└── " if last else "├── ") + key)
            if node[key] is not None:
                walk(node[key], prefix + ("    " if last else "│   "))

    walk(tree, "")
    return "\n".join(lines)


def scan_project(root: Path, include_tests: bool) -> ScanResult:
    files: list[SourceFile] = []
    skipped: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS
            and not d.endswith(".egg-info")
            and (include_tests or d != "tests")
        ]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() not in INCLUDED_EXTENSIONS and name not in INCLUDED_EXACT_NAMES:
                continue
            rel = path.relative_to(root).as_posix()
            try:
                if path.resolve() == SELF_PATH:
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped.append(f"{rel} (too large)")
                    continue
                raw = path.read_bytes()
            except OSError:
                skipped.append(f"{rel} (unreadable)")
                continue
            if b"\x00" in raw:
                skipped.append(f"{rel} (binary)")
                continue
            content = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
            if name == "__init__.py" and not content.strip():
                continue
            files.append(SourceFile(rel, content.rstrip()))

    files.sort(key=lambda f: _sort_key(f.rel))
    return ScanResult(
        root=root,
        files=files,
        tree=_build_tree(root.name, [f.rel for f in files]),
        total_lines=sum(f.content.count("\n") + 1 for f in files),
        skipped=skipped,
    )


# ─────────────────────────────── Formatting ──────────────────────────────────

def _fence(text: str) -> str:
    """A code fence longer than any backtick run inside the text, so it never breaks."""
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=0)
    return "`" * max(3, longest + 1)


def _lang(rel: str) -> str:
    name = rel.rsplit("/", 1)[-1]
    if name == ".gitignore":
        return "gitignore"
    return LANG_BY_SUFFIX.get(Path(name).suffix.lower(), "")


_TITLES = {
    "whole": "### FILE: {rel}",
    "first": "### FILE: {rel}  [this file is split - it continues in the next part]",
    "middle": "### FILE: {rel}  [continued from the previous part - continues in the next part]",
    "last": "### FILE: {rel}  [continued from the previous part - end of file]",
}
LONG_LINE = 2_000   # longer lines (minified data etc.) are wrapped so they can be split
MIN_PIECE = 1_500   # don't leave tiny file fragments at the end of a part


class _PreparedFile:
    """A file broken into lines with prefix sums, so any line range's size is O(1)."""

    def __init__(self, f: SourceFile) -> None:
        self.rel = f.rel
        lines: list[str] = []
        for line in f.content.split("\n"):
            while ulen(line) > LONG_LINE:
                cut = LONG_LINE // 2 if ulen(line) != len(line) else LONG_LINE
                lines.append(line[:cut])
                line = line[cut:]
            lines.append(line)
        self.lines = lines
        self.prefix = [0]
        for line in lines:
            self.prefix.append(self.prefix[-1] + ulen(line) + 1)
        self.fence = _fence(f.content)
        self.lang = _lang(f.rel)
        fixed = 2 * len(self.fence) + len(self.lang) + 2  # fence+lang, newline, newline, fence
        self.title_len = {k: ulen(t.format(rel=self.rel)) + 1 + fixed for k, t in _TITLES.items()}

    def size(self, kind: str, start: int, end: int) -> int:
        return self.title_len[kind] + (self.prefix[end] - self.prefix[start] - 1)

    def render(self, kind: str, start: int, end: int) -> str:
        title = _TITLES[kind].format(rel=self.rel)
        body = "\n".join(self.lines[start:end])
        return f"{title}\n{self.fence}{self.lang}\n{body}\n{self.fence}"


_ANALYZE_REQUEST = (
    "1. Analyze the whole codebase: its structure, how the modules interact, the main logic and data flow.\n"
    "2. Give me a short summary of your understanding.\n"
    "3. Confirm that you are ready to help me write, debug and improve this code, "
    "and that you will help me code it from now on."
)


def _wrap(idx: int, total: int, body: str, tree_section: str) -> str:
    """Put the START line, instructions for Gemini, and the END line around one part."""
    n = idx + 1
    start_line = f"===== PART {n} OF {total} - START ====="
    end_line = f"===== PART {n} OF {total} - END ====="
    cut_check = (
        f'First check that this message ends with the line "{end_line}". '
        f'If that line is missing, the paste was cut off: reply ONLY "PART {n} WAS CUT OFF" and nothing else.\n'
    )

    if total == 1:
        header = (
            f"{start_line}\n"
            "This is my complete project codebase in ONE message: the directory structure and all source code.\n"
            f"{cut_check}\n"
            f"{tree_section}"
            "=== CODE STARTS HERE ===\n\n"
        )
        footer = (
            "\n\n=== CODE ENDS HERE - THIS IS THE WHOLE PROJECT ===\n"
            "Please:\n"
            f"{_ANALYZE_REQUEST}\n"
            f"{end_line}"
        )
    elif n == 1:
        header = (
            f"{start_line}\n"
            f"My project is too large for one message, so I will send it in {total} parts, in order.\n"
            f"This is PART 1 of {total}: the project structure and the first files. It is ONLY a part of the code.\n"
            f"{cut_check}"
            "Do NOT analyze, summarize or write any code yet.\n"
            f'If the part is complete, reply ONLY: "Part 1/{total} received - ready for part 2." and wait.\n\n'
            f"{tree_section}"
            f"=== CODE STARTS HERE (PART 1 OF {total}) ===\n\n"
        )
        footer = (
            f"\n\n=== CODE PAUSES HERE - IT CONTINUES IN PART 2 OF {total} ===\n"
            f'Reply ONLY: "Part 1/{total} received - ready for part 2."\n'
            f"{end_line}"
        )
    elif n < total:
        header = (
            f"{start_line}\n"
            f"This is PART {n} of {total} of my project. It continues directly from part {n - 1}. "
            "It is ONLY a part of the code.\n"
            f"{cut_check}"
            "Do NOT analyze, summarize or write any code yet.\n"
            f'If the part is complete, reply ONLY: "Part {n}/{total} received - ready for part {n + 1}." and wait.\n\n'
            f"=== CODE CONTINUES HERE (PART {n} OF {total}) ===\n\n"
        )
        footer = (
            f"\n\n=== CODE PAUSES HERE - IT CONTINUES IN PART {n + 1} OF {total} ===\n"
            f'Reply ONLY: "Part {n}/{total} received - ready for part {n + 1}."\n'
            f"{end_line}"
        )
    else:
        header = (
            f"{start_line}\n"
            f"This is the FINAL part ({n} of {total}) of my project. It continues directly from part {n - 1}.\n"
            f"{cut_check}\n"
            f"=== CODE CONTINUES HERE (PART {n} OF {total}) ===\n\n"
        )
        footer = (
            f"\n\n=== CODE ENDS HERE - ALL {total} PARTS ARE SENT ===\n"
            f"You now have my complete project (parts 1 to {total}). "
            f"If you did not receive all {total} parts, tell me which ones are missing. Otherwise, please:\n"
            f"{_ANALYZE_REQUEST}\n"
            f"{end_line}"
        )
    return f"{header}{body}{footer}"


# ─────────────────────────────── Splitting ───────────────────────────────────

class SplitError(Exception):
    pass


def _pack(files: list[_PreparedFile], budgets: list[int]) -> list[list[str]] | None:
    """
    Fill parts in order, as full as their budget allows. A file that doesn't fit
    in the space left continues in the next part (cut at a line boundary).
    Returns the blocks of each part, or None if everything doesn't fit in len(budgets) parts.
    """
    parts: list[list[str]] = [[]]
    used = 0

    def room() -> int:
        return budgets[len(parts) - 1] - used - (len(SEP) if parts[-1] else 0)

    for f in files:
        start, n = 0, len(f.lines)
        while start < n:
            kind_rest = "whole" if start == 0 else "last"
            if f.size(kind_rest, start, n) <= room():
                block_size = f.size(kind_rest, start, n)
                used += block_size + (len(SEP) if parts[-1] else 0)
                parts[-1].append(f.render(kind_rest, start, n))
                break

            # How many lines fit into this part?
            kind = "first" if start == 0 else "middle"
            space = room()
            lo, hi = start, n - 1  # leave at least one line for the next piece
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if f.size(kind, start, mid) <= space:
                    lo = mid
                else:
                    hi = mid - 1
            end = lo
            if end > start and (f.size(kind, start, end) >= MIN_PIECE or not parts[-1]):
                parts[-1].append(f.render(kind, start, end))
                start = end
            elif not parts[-1]:
                return None  # a single line doesn't fit an empty part

            if len(parts) == len(budgets):
                return None
            parts.append([])
            used = 0
    return parts


def split_into_parts(scan: ScanResult, limit: int) -> list[str]:
    """Fewest parts that fit `limit`, then the most even split for that number of parts."""
    if not scan.files:
        raise SplitError("No matching files found in this folder.")

    tf = _fence(scan.tree)
    tree_section = f"PROJECT STRUCTURE ({len(scan.files)} files):\n{tf}text\n{scan.tree}\n{tf}\n\n"
    first_overhead = ulen(_wrap(0, 999, "", tree_section))
    if limit - first_overhead < 2 * LONG_LINE + 500:
        raise SplitError(
            f"The limit of {limit:,} is too small: the instructions and project tree alone need "
            f"{first_overhead:,} characters. Use at least {first_overhead + 2 * LONG_LINE + 500:,}."
        )

    files = [_PreparedFile(f) for f in scan.files]
    overhead_cache: dict[int, list[int]] = {}

    def budgets(k: int, cap: int) -> list[int]:
        if k not in overhead_cache:
            overhead_cache[k] = [ulen(_wrap(j, k, "", tree_section)) for j in range(k)]
        return [cap - o for o in overhead_cache[k]]

    # 1) fewest parts that fit under your limit
    k = 1
    while _pack(files, budgets(k, limit)) is None:
        k += 1
        if k > 10_000:
            raise SplitError("Could not split the code - try a larger limit.")

    # 2) smallest part size that still works with k parts -> parts come out evenly sized
    lo, hi = 1, limit
    while lo < hi:
        mid = (lo + hi) // 2
        if _pack(files, budgets(k, mid)) is not None:
            hi = mid
        else:
            lo = mid + 1

    packed = _pack(files, budgets(k, lo)) or _pack(files, budgets(k, limit))
    packed = [p for p in packed if p]
    total = len(packed)
    parts = [_wrap(j, total, SEP.join(blocks), tree_section) for j, blocks in enumerate(packed)]

    biggest = max(ulen(p) for p in parts)
    if biggest > limit:
        raise SplitError(f"Internal check failed: a part has {biggest:,} > {limit:,} characters. Try a slightly lower limit.")
    return parts


# ──────────────────────────────── Settings ───────────────────────────────────

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# ─────────────────────────────────── UI ──────────────────────────────────────

BG, CARD, FIELD, BORDER = "#1e1e2e", "#282839", "#313244", "#45475a"
TEXT, MUTED, ACCENT, ACCENT_HOVER = "#cdd6f4", "#a6adc8", "#89b4fa", "#74c7ec"
GREEN, YELLOW, RED, DARK = "#a6e3a1", "#f9e2af", "#f38ba8", "#11111b"
UI_FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO_FONT = "Consolas" if sys.platform == "win32" else "Courier"
PART_BUTTON_COLUMNS = 6


class CodebaseCopierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings = load_settings()

        self.project_root = SELF_PATH.parent  # the folder this script is in

        self.scan: ScanResult | None = None
        self.parts: list[str] = []
        self.copied: set[int] = set()
        self.last_copied = -1
        self.part_buttons: list[tk.Button] = []
        self._resplit_job: str | None = None

        self._setup_style()
        self._build_ui()

        model = self.settings.get("model")
        self.model_var.set(model if model in GEMINI_MODELS else DEFAULT_MODEL)
        self._on_model_changed(restore_plan=self.settings.get("plan"))
        self.limit_var.trace_add("write", lambda *_: self._on_limit_typed())
        self._rescan()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- layout ----------

    def _setup_style(self) -> None:
        self.root.title("Gemini Codebase Copier")
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "TCombobox", fieldbackground=FIELD, background=BORDER, foreground=TEXT,
            arrowcolor=TEXT, bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, padding=4,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)],
        )
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", DARK)
        self.root.option_add("*TCombobox*Listbox.font", (UI_FONT, 10))

    def _card(self, title: str, expand: bool = False) -> tk.Frame:
        card = tk.Frame(self.root, bg=CARD, padx=16, pady=12, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="both" if expand else "x", expand=expand, padx=18, pady=(0, 10))
        tk.Label(card, text=title, font=(UI_FONT, 11, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 8))
        return card

    def _small_button(self, parent: tk.Widget, text: str, command) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, font=(UI_FONT, 9), bg=FIELD, fg=TEXT,
            activebackground=BORDER, activeforeground=TEXT, relief="flat", padx=10, pady=3, cursor="hand2",
        )

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg=BG, padx=18, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="Gemini Codebase Copier", font=(UI_FONT, 16, "bold"), bg=BG, fg=ACCENT).pack(anchor="w")
        row = tk.Frame(header, bg=BG)
        row.pack(fill="x", pady=(4, 0))
        self.folder_label = tk.Label(row, text="", font=(UI_FONT, 9), bg=BG, fg=MUTED, anchor="w")
        self.folder_label.pack(side="left", fill="x", expand=True)
        self._small_button(row, "Rescan files", self._rescan).pack(side="right")
        self._small_button(row, "Change folder…", self._choose_folder).pack(side="right", padx=(0, 6))

        # Step 1
        cfg = self._card("1. Choose your Gemini model and Google AI plan")
        grid = tk.Frame(cfg, bg=CARD)
        grid.pack(fill="x")
        for col, label in enumerate(("Gemini model", "Your Google AI plan", "Max characters per message")):
            tk.Label(grid, text=label, font=(UI_FONT, 9), bg=CARD, fg=MUTED).grid(row=0, column=col, sticky="w", padx=(0, 12))

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(grid, textvariable=self.model_var, values=list(GEMINI_MODELS), state="readonly", width=32, font=(UI_FONT, 10))
        self.model_combo.grid(row=1, column=0, sticky="we", padx=(0, 12), pady=(2, 0))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_changed())

        self.plan_var = tk.StringVar()
        self.plan_combo = ttk.Combobox(grid, textvariable=self.plan_var, state="readonly", width=24, font=(UI_FONT, 10))
        self.plan_combo.grid(row=1, column=1, sticky="we", padx=(0, 12), pady=(2, 0))
        self.plan_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_plan_changed())

        self.limit_var = tk.StringVar()
        tk.Spinbox(
            grid, from_=5_000, to=2_000_000, increment=1_000, textvariable=self.limit_var, width=12,
            font=(UI_FONT, 10), bg=FIELD, fg=TEXT, insertbackground=TEXT, buttonbackground=BORDER,
            relief="flat", highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        ).grid(row=1, column=2, sticky="we", pady=(2, 0), ipady=3)
        grid.columnconfigure(0, weight=3)
        grid.columnconfigure(1, weight=3)
        grid.columnconfigure(2, weight=1)

        opts = tk.Frame(cfg, bg=CARD)
        opts.pack(fill="x", pady=(10, 0))
        self.include_tests_var = tk.BooleanVar(value=self.settings.get("include_tests", True))
        tk.Checkbutton(
            opts, text="Include the tests/ folder", variable=self.include_tests_var, command=self._on_tests_toggled,
            bg=CARD, fg=TEXT, selectcolor=FIELD, activebackground=CARD, activeforeground=TEXT, font=(UI_FONT, 10),
        ).pack(side="left")
        self.limit_hint = tk.Label(opts, text="", font=(UI_FONT, 9), bg=CARD, fg=MUTED)
        self.limit_hint.pack(side="right")

        # Step 2
        act = self._card("2. Copy each part, paste it into the chat, send - in order")
        self.summary_label = tk.Label(act, text="", font=(UI_FONT, 11, "bold"), bg=CARD, fg=GREEN, anchor="w", justify="left")
        self.summary_label.pack(fill="x")
        self.detail_label = tk.Label(act, text="", font=(UI_FONT, 9), bg=CARD, fg=MUTED, anchor="w", justify="left", wraplength=760)
        self.detail_label.pack(fill="x", pady=(2, 10))

        self.copy_btn = tk.Button(
            act, text="", command=self._copy_next, font=(UI_FONT, 12, "bold"), bg=ACCENT, fg=DARK,
            activebackground=ACCENT_HOVER, activeforeground=DARK, relief="flat", pady=10, cursor="hand2",
        )
        self.copy_btn.pack(fill="x")

        self.parts_frame = tk.Frame(act, bg=CARD)
        self.parts_frame.pack(fill="x", pady=(10, 0))

        bottom = tk.Frame(act, bg=CARD)
        bottom.pack(fill="x", pady=(10, 0))
        self.status_label = tk.Label(bottom, text="", font=(UI_FONT, 10, "bold"), bg=CARD, fg=YELLOW, anchor="w", justify="left", wraplength=640)
        self.status_label.pack(side="left", fill="x", expand=True)
        self._small_button(bottom, "Start over", self._reset_progress).pack(side="right")

        # Step 3
        prev = self._card("Preview", expand=True)
        self.preview_title = prev.winfo_children()[0]
        self.preview = scrolledtext.ScrolledText(
            prev, wrap="word", font=(MONO_FONT, 9), bg="#181825", fg=TEXT, insertbackground=TEXT,
            relief="flat", height=14, padx=8, pady=6,
        )
        self.preview.pack(fill="both", expand=True)

        self.root.update_idletasks()
        self.root.minsize(760, 620)

    # ---------- model / plan / limit ----------

    def _key(self) -> str:
        return f"{self.model_var.get()}|{self.plan_var.get()}"

    def _preset_limit(self) -> int:
        return GEMINI_MODELS.get(self.model_var.get(), {}).get("chars", SAFE_MESSAGE_CHARS)

    def _context_tokens(self) -> int:
        return GEMINI_PLANS.get(self.plan_var.get(), 32_000)

    def _on_model_changed(self, restore_plan: str | None = None) -> None:
        allowed = GEMINI_MODELS.get(self.model_var.get(), {}).get("plans")
        plans = [p for p in GEMINI_PLANS if allowed is None or p in allowed]
        self.plan_combo["values"] = plans
        wanted = restore_plan or self.plan_var.get()
        default = "Google AI Pro (incl. Student)" if "Google AI Pro (incl. Student)" in plans else (plans[0] if plans else "")
        self.plan_var.set(wanted if wanted in plans else default)
        self._on_plan_changed()

    def _on_plan_changed(self) -> None:
        overrides = self.settings.get("limit_overrides", {})
        new_value = str(overrides.get(self._key(), self._preset_limit()))
        if self.limit_var.get() == new_value:
            self._schedule_resplit()  # same limit, but the memory warning may change
        self.limit_var.set(new_value)
        self._save()

    def _read_limit(self) -> int | None:
        try:
            value = int(self.limit_var.get().replace(",", "").replace(" ", "").replace("_", ""))
        except ValueError:
            return None
        return value if value >= 2_000 else None

    def _on_limit_typed(self) -> None:
        limit = self._read_limit()
        preset = self._preset_limit()
        if limit is None:
            self.limit_hint.config(text="Enter a whole number (at least 2,000)", fg=RED)
            return
        overrides = self.settings.setdefault("limit_overrides", {})
        if limit == preset:
            overrides.pop(self._key(), None)
            self.limit_hint.config(text=f"Researched max for this mode: {preset:,} characters", fg=MUTED)
        else:
            overrides[self._key()] = limit
            self.limit_hint.config(text=f"Custom value - researched max is {preset:,}", fg=YELLOW)
        self._save()
        self._schedule_resplit()

    def _schedule_resplit(self) -> None:
        if self._resplit_job:
            self.root.after_cancel(self._resplit_job)
        self._resplit_job = self.root.after(350, self._resplit)

    def _on_tests_toggled(self) -> None:
        self._save()
        self._rescan()

    def _choose_folder(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(self.project_root), title="Choose your project folder")
        if chosen:
            self.project_root = Path(chosen)
            self._rescan()

    # ---------- scanning / splitting ----------

    def _rescan(self) -> None:
        self.folder_label.config(text=f"Project: {self.project_root}")
        self.root.title(f"Gemini Codebase Copier - {self.project_root.name}")
        self.scan = scan_project(self.project_root, self.include_tests_var.get())
        self._resplit()

    def _resplit(self) -> None:
        self._resplit_job = None
        limit = self._read_limit()
        if self.scan is None or limit is None:
            return
        try:
            self.parts = split_into_parts(self.scan, limit)
        except SplitError as exc:
            self.parts = []
            self._reset_progress()
            self.summary_label.config(text="Cannot split", fg=RED)
            self.detail_label.config(text=str(exc))
            self._set_status("", YELLOW)
            return

        sizes = [ulen(p) for p in self.parts]
        n = len(self.parts)
        self.summary_label.config(
            text=f"{n} part{'s' if n != 1 else ''} to copy   ·   {len(self.scan.files)} files   ·   "
                 f"{self.scan.total_lines:,} lines   ·   {sum(sizes):,} characters",
            fg=GREEN,
        )
        detail = f"Each part fits your limit of {limit:,} characters, including the instructions."
        if n > 1:
            detail += f"  Part sizes: {min(sizes):,} to {max(sizes):,}."
        tokens = sum(estimate_tokens(p) for p in self.parts)
        context = self._context_tokens()
        self.detail_label.config(fg=MUTED)
        if tokens > context * CONTEXT_WARN_SHARE:
            fix = "Upgrade your plan, untick" if context < 1_000_000 else "Untick"
            detail += (
                f"\n⚠ The whole project is about {tokens:,} tokens, but Gemini on this plan remembers only "
                f"{context:,} tokens per chat - it would forget the first parts. "
                f"{fix} the tests folder, or send fewer files."
            )
            self.detail_label.config(fg=RED)
        else:
            detail += (
                f"\nAbout {tokens:,} tokens - fits in Gemini's memory for this plan ({context:,} tokens; "
                f"room for about {int(context * CONTEXT_WARN_SHARE * CHARS_PER_TOKEN):,} characters of code)."
            )
        if self.scan.skipped:
            shown = ", ".join(self.scan.skipped[:4]) + (" …" if len(self.scan.skipped) > 4 else "")
            detail += f"\nSkipped {len(self.scan.skipped)} file(s): {shown}"
        self.detail_label.config(text=detail)
        self._reset_progress()

    # ---------- copying ----------

    def _ai_name(self) -> str:
        return "Gemini"

    def _next_index(self) -> int | None:
        n = len(self.parts)
        for i in list(range(self.last_copied + 1, n)) + list(range(0, n)):
            if i not in self.copied:
                return i
        return None

    def _reset_progress(self) -> None:
        self.copied.clear()
        self.last_copied = -1
        self._rebuild_part_buttons()
        self._refresh()
        if self.parts:
            ai = self._ai_name()
            self._set_status(f"Ready. Open a NEW chat with {ai}, then click the blue button to copy Part 1.", YELLOW)

    def _copy_next(self) -> None:
        if not self.parts:
            return
        if len(self.copied) == len(self.parts):
            self._reset_progress()
        if not self.copied:
            self._rescan()  # pick up any edits you made since the window opened
            if not self.parts:
                return
        idx = self._next_index()
        if idx is not None:
            self._copy(idx)

    def _copy(self, idx: int) -> None:
        text = self.parts[idx]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

        expected = self._next_index()
        self.copied.add(idx)
        self.last_copied = idx
        n, total, ai = idx + 1, len(self.parts), self._ai_name()

        if total == 1:
            msg = f"Copied! Paste it into {ai} and send - this is the whole project."
            color = GREEN
        elif n == total:
            msg = f"Copied the FINAL part {n}/{total}. Paste and send - {ai} will now analyze the whole project."
            color = GREEN
        else:
            msg = (f"Copied part {n}/{total}. Paste it into {ai} and send. When it replies "
                   f"\"Part {n}/{total} received\", click the button again for part {n + 1}.")
            color = ACCENT
        missing = [str(i + 1) for i in range(total) if i not in self.copied]
        if n == total and missing:
            msg += f"  Note: you haven't copied part(s) {', '.join(missing)} yet."
            color = YELLOW
        elif expected is not None and idx != expected and total > 1:
            msg += f"  (You skipped ahead - the next part in order was {expected + 1}.)"
            color = YELLOW
        self._set_status(msg, color)
        self._refresh()

    # ---------- view updates ----------

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.config(text=text, fg=color)

    def _rebuild_part_buttons(self) -> None:
        for b in self.part_buttons:
            b.destroy()
        self.part_buttons = []
        if len(self.parts) <= 1:
            return
        for i in range(len(self.parts)):
            b = tk.Button(
                self.parts_frame, command=lambda i=i: self._copy(i), font=(UI_FONT, 9, "bold"),
                relief="flat", padx=8, pady=4, cursor="hand2", activebackground=BORDER, activeforeground=TEXT,
            )
            b.grid(row=i // PART_BUTTON_COLUMNS, column=i % PART_BUTTON_COLUMNS, sticky="we", padx=(0, 6), pady=(0, 6))
            self.part_buttons.append(b)
        for c in range(PART_BUTTON_COLUMNS):
            self.parts_frame.columnconfigure(c, weight=1, uniform="parts")

    def _refresh(self) -> None:
        total = len(self.parts)
        if total == 0:
            self.copy_btn.config(text="Nothing to copy", state="disabled", bg=FIELD)
            self._show_preview(None)
            return

        for i, b in enumerate(self.part_buttons):
            done = i in self.copied
            b.config(
                text=f"{'✓ ' if done else ''}Part {i + 1}  ({ulen(self.parts[i]) // 1000}k)",
                bg=GREEN if done else FIELD, fg=DARK if done else TEXT,
            )

        nxt = self._next_index()
        if nxt is None:
            self.copy_btn.config(text=f"✓ All {total} parts copied - click to start over", state="normal", bg=GREEN, activebackground=GREEN)
            self._show_preview(self.last_copied)
            return
        role = "WHOLE PROJECT" if total == 1 else "START" if nxt == 0 else "FINAL" if nxt == total - 1 else "MIDDLE"
        self.copy_btn.config(
            text=f"Copy part {nxt + 1} of {total}   [{role}]   ({ulen(self.parts[nxt]):,} characters)",
            state="normal", bg=ACCENT, activebackground=ACCENT_HOVER,
        )
        self._show_preview(nxt)

    def _show_preview(self, idx: int | None) -> None:
        self.preview.config(state="normal")
        self.preview.delete("1.0", "end")
        if idx is None or not (0 <= idx < len(self.parts)):
            self.preview_title.config(text="Preview")
        else:
            text = self.parts[idx]
            self.preview_title.config(text=f"Preview - part {idx + 1} of {len(self.parts)} (start and end shown)")
            if len(text) <= 6_000:
                self.preview.insert("1.0", text)
            else:
                hidden = len(text) - 4_500
                self.preview.insert("1.0", f"{text[:3_000]}\n\n· · · [{hidden:,} characters not shown] · · ·\n\n{text[-1_500:]}")
        self.preview.config(state="disabled")

    # ---------- persistence ----------

    def _save(self) -> None:
        self.settings.update(
            model=self.model_var.get(),
            plan=self.plan_var.get(),
            include_tests=bool(self.include_tests_var.get()),
        )
        save_settings(self.settings)

    def _on_close(self) -> None:
        self._save()
        self.root.destroy()


def main() -> None:
    if sys.platform == "win32":
        try:  # sharp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    CodebaseCopierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()