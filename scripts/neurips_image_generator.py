"""LaTeX/TikZ figure generator driven by LLMs.

Generates publication-quality figures by asking an LLM to write a standalone
TikZ/LaTeX document, compiles it with latexmk, and iteratively feeds any
compilation errors back to the model until it builds
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


SCRIPT_DIR = Path(__file__).resolve().parent


API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

CHAT_BASE = os.getenv("CHAT_ENDPOINT", "").rstrip("/").rstrip('"')
ANTHROPIC_BASE = os.getenv("REASONING_ENDPOINT", "").rstrip("/").rstrip('"')

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
REASONING_MODEL = os.getenv("REASONING_MODEL", "claude-opus-4-6")

chat_client = OpenAI(api_key=API_KEY, base_url=CHAT_BASE) if CHAT_BASE else None

COST_TABLE = {
    "gpt-5-mini":      {"input_per_1k": 0.0004, "output_per_1k": 0.0016},
    "claude-opus-4-6": {"input_per_1k": 0.015,  "output_per_1k": 0.075},
}


# LLM system prompts

TIKZ_SYSTEM = r"""You are an expert at writing publication-quality figures in TikZ/LaTeX
for academic papers (NeurIPS, ICML, CVPR style).

Given a text description of a figure, produce a COMPLETE, SELF-CONTAINED
LaTeX document that renders the figure using TikZ/pgfplots.

Hard requirements:
- Start with \documentclass[border=5pt]{standalone}
- Only load widely-available packages: tikz, pgfplots, xcolor, amsmath,
  amssymb, amsfonts, bm. Do NOT load fontawesome, emoji, or exotic packages.
  If the description mentions icons, draw them with TikZ primitives.
- Use \pgfplotsset{compat=1.18} when loading pgfplots.
- Define named colors near the top with \definecolor using hex values.
- Every text label must appear EXACTLY as specified in the description.
  No paraphrasing, no abbreviations, no placeholder text.
- Flat, clean, academic style: no gradients, no 3D effects, no drop shadows
  unless explicitly requested.
- Choose coordinates and sizes so the layout matches the description.
- For plots, use pgfplots with inline coordinates, do not read external files.
- The document MUST compile cleanly with pdflatex.

Output format:
Return ONLY the complete LaTeX source inside a single ```latex ... ``` code
fence. No commentary before or after the fence.
"""

FIX_SYSTEM = r"""You are debugging a TikZ/LaTeX compilation failure.

You will receive:
1. The current LaTeX source.
2. The relevant portion of the pdflatex error log.

Fix ALL errors and return the complete corrected source. Preserve the original
figure content and layout, change only what is needed to make it compile.
Keep the \documentclass[border=5pt]{standalone} structure.

Output format:
Return ONLY the complete corrected LaTeX inside a single ```latex ... ``` code
fence. No commentary.
"""

REVIEW_SYSTEM = r"""You are reviewing a rendered academic figure against its written specification.

You will receive:
1. The original text description.
2. The current TikZ/LaTeX source.
3. A PNG rendering of the compiled figure.

Check whether the rendering matches the description: zones present, labels
correct, colors right, alignment sensible, nothing overlapping, proportions
readable. If it matches well, respond with exactly the single word: LGTM

If it needs fixes, return the complete revised LaTeX source inside a
```latex ... ``` code fence. Change only what is needed. Keep the
\documentclass[border=5pt]{standalone} structure.
"""


# Cost tracking

class CostTracker:
    """Accumulates estimated LLM costs across all calls."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add_chat(self, model: str, input_tokens: int, output_tokens: int, label: str = "") -> None:
        table = COST_TABLE.get(model, {})
        cost = (
            (input_tokens / 1000) * table.get("input_per_1k", 0)
            + (output_tokens / 1000) * table.get("output_per_1k", 0)
        )
        self.entries.append({
            "model": model,
            "label": label,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        })

    @property
    def total(self) -> float:
        return sum(e["cost"] for e in self.entries)

    def summary(self) -> str:
        lines = ["\n--- Cost Summary ---"]
        for i, e in enumerate(self.entries, 1):
            label = f" [{e['label']}]" if e["label"] else ""
            lines.append(
                f"  [{i}] {e['model']}{label} "
                f"({e['input_tokens']} in / {e['output_tokens']} out): "
                f"${e['cost']:.4f}"
            )
        lines.append(f"  TOTAL estimated cost: ${self.total:.4f}")
        return "\n".join(lines)


# Prompt / color helpers

def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_colors(path: Path | None) -> dict | None:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def format_color_block(colors: dict) -> str:
    lines = ["", "----- COLOR PALETTE (use these exact hex values) -----"]
    for section, mapping in colors.items():
        lines.append(f"\n{section.upper()}:")
        if isinstance(mapping, dict):
            for name, val in mapping.items():
                if isinstance(val, dict):
                    parts = ", ".join(f"{k}={v}" for k, v in val.items())
                    lines.append(f"  - {name}: {parts}")
                else:
                    lines.append(f"  - {name}: {val}")
    return "\n".join(lines)


def inject_colors(prompt: str, colors: dict | None) -> str:
    if colors is None:
        return prompt
    return f"{prompt}\n\n{format_color_block(colors)}"


def _extract_latex(text: str) -> str:
    """Pull a LaTeX source out of a model response (handles ```latex fences)."""
    m = re.search(r"```(?:latex|tex)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_error_excerpt(log: str, context_lines: int = 10) -> str:
    """Keep the interesting chunks of a pdflatex log, not the whole thing."""
    lines = log.splitlines()
    keep: list[str] = []
    for i, line in enumerate(lines):
        if line.startswith("!") or " Error" in line or line.startswith("l."):
            start = max(0, i - 2)
            end = min(len(lines), i + context_lines + 1)
            keep.append("\n".join(lines[start:end]))
            keep.append("---")
    if not keep:
        return "\n".join(lines[-60:])
    excerpt = "\n".join(keep)
    if len(excerpt) > 6000:
        excerpt = excerpt[:6000] + "\n... [truncated]"
    return excerpt


# LLM calls

def call_gpt(system: str, user_content, tracker: CostTracker, label: str) -> str:
    if chat_client is None:
        raise RuntimeError("CHAT_ENDPOINT is not configured.")
    resp = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=8192)
    msg = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage:
        tracker.add_chat(CHAT_MODEL, usage.prompt_tokens, usage.completion_tokens, label)
    return msg


def call_claude(system: str, user_content, tracker: CostTracker, label: str) -> str:
    if not ANTHROPIC_BASE:
        raise RuntimeError("REASONING_ENDPOINT is not configured.")
    r = requests.post(
        f"{ANTHROPIC_BASE}/messages",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": REASONING_MODEL,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        },
        timeout=300)
    if not r.ok:
        raise RuntimeError(f"Claude call failed: {r.status_code} {r.text[:500]}")
    body = r.json()
    msg = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            msg += block.get("text", "")
    usage = body.get("usage", {})
    tracker.add_chat(
        REASONING_MODEL,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        label)
    return msg


def call_llm(agent: str, system: str, user_content, tracker: CostTracker, label: str) -> str:
    if agent == "claude":
        return call_claude(system, user_content, tracker, label)
    return call_gpt(system, user_content, tracker, label)


# LaTeX compilation

def compile_latex(tex_path: Path) -> tuple[bool, str, Path | None]:
    """Compile a .tex file with latexmk. Returns (ok, log_text, pdf_path)."""
    workdir = tex_path.parent
    name = tex_path.stem

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError(
            "latexmk not found on PATH. Install a LaTeX distribution "
            "(MiKTeX or TeX Live) so latexmk/pdflatex are available."
        )

    try:
        proc = subprocess.run(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=240)
    except subprocess.TimeoutExpired as e:
        return False, f"latexmk timed out: {e}", None

    log_file = workdir / f"{name}.log"
    if log_file.exists():
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
    else:
        log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    pdf_path = workdir / f"{name}.pdf"
    ok = proc.returncode == 0 and pdf_path.exists()
    return ok, log_text, (pdf_path if ok else None)


# PDF -> PNG rasterization (optional, requires pymupdf)

def pdf_to_png(pdf_path: Path, png_path: Path, dpi: int = 200) -> bool:
    try:
        import fitz  # pymupdf
    except ImportError:
        return False
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(png_path)
        doc.close()
        return True
    except Exception as e:
        print(f"[render] PDF->PNG failed: {e}")
        return False


# Pipeline stages

def generate_tikz(prompt: str, agent: str, tracker: CostTracker) -> str:
    print(f"[tikz] Generating standalone LaTeX with {agent} ...")
    user = f"Figure description:\n\n{prompt}"
    resp = call_llm(agent, TIKZ_SYSTEM, user, tracker, "generate")
    return _extract_latex(resp)


def fix_tikz_errors(tex: str, error_log: str, agent: str, tracker: CostTracker) -> str:
    excerpt = _extract_error_excerpt(error_log)
    user = (
        f"Current LaTeX source:\n\n```latex\n{tex}\n```\n\n"
        f"Compilation error log (excerpt):\n\n```\n{excerpt}\n```"
    )
    resp = call_llm(agent, FIX_SYSTEM, user, tracker, "fix-compile")
    return _extract_latex(resp)


def visual_review(
    prompt: str,
    tex: str,
    png_path: Path,
    agent: str,
    tracker: CostTracker) -> tuple[bool, str]:
    """Ask the LLM whether the rendered figure matches the description.

    Returns (needs_fix, response_or_new_tex).
    """
    img_b64 = base64.b64encode(png_path.read_bytes()).decode()
    text_part = (
        f"Original description:\n\n{prompt}\n\n"
        f"Current LaTeX source:\n\n```latex\n{tex}\n```\n\n"
        "Rendered figure attached."
    )

    if agent == "claude":
        user_content = [
            {"type": "text", "text": text_part},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                },
            },
        ]
    else:
        user_content = [
            {"type": "text", "text": text_part},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ]

    resp = call_llm(agent, REVIEW_SYSTEM, user_content, tracker, "visual-review")
    if resp.strip().upper().startswith("LGTM"):
        return False, resp

    revised = _extract_latex(resp)
    if not revised or not revised.lstrip().startswith("\\documentclass"):
        # Model didn't return a proper source, treat as a no-op.
        return False, resp
    return True, revised


# Main pipeline

def run(args: argparse.Namespace) -> None:
    tracker = CostTracker()
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_prompt = load_prompt(prompt_path)
    colors = load_colors(Path(args.colors)) if args.colors else None
    prompt = inject_colors(raw_prompt, colors)

    (outdir / "prompt.txt").write_text(prompt, encoding="utf-8")

    print(f"[init] Prompt: {prompt_path} ({len(prompt)} chars)")
    print(f"[init] Outdir: {outdir}")
    print(f"[init] Agent : {args.agent}  (compile retries: {args.max_compile_retries})")
    if args.visual_review:
        print(f"[init] Visual review: on  (iterations: {args.visual_iterations})")
    print("---")

    # Step 1: generate initial TikZ
    tex = generate_tikz(prompt, args.agent, tracker)

    tex_path = outdir / "figure.tex"
    tex_path.write_text(tex, encoding="utf-8")

    # Step 2: compile, feeding errors back to the LLM if needed
    compile_ok = False
    pdf_path: Path | None = None
    last_log = ""
    for attempt in range(1, args.max_compile_retries + 2):
        print(f"\n[compile] Attempt {attempt} ...")
        (outdir / f"figure_attempt_{attempt}.tex").write_text(tex, encoding="utf-8")
        tex_path.write_text(tex, encoding="utf-8")
        ok, log, pdf = compile_latex(tex_path)
        last_log = log
        if ok:
            print("[compile] SUCCESS")
            compile_ok = True
            pdf_path = pdf
            break
        print(f"[compile] FAILED (attempt {attempt})")
        if attempt > args.max_compile_retries:
            print("[compile] No retries left.")
            break
        print("[compile] Asking LLM to fix compilation errors ...")
        tex = fix_tikz_errors(tex, log, args.agent, tracker)

    (outdir / "compile_log.txt").write_text(last_log, encoding="utf-8")

    if not compile_ok:
        print("\n[done] Compilation never succeeded. See compile_log.txt and figure.tex.")
        _write_metadata(outdir, args, prompt_path, tracker, compiled=False, visual_reviewed=False)
        print(tracker.summary())
        sys.exit(2)

    # Optional PNG preview (required for visual review)
    png_path: Path | None = None
    if args.visual_review or args.png:
        png_path = outdir / "figure.png"
        if not pdf_to_png(pdf_path, png_path):
            print("[render] pymupdf not available, skipping PNG preview.")
            print("         Install it with: pip install pymupdf")
            png_path = None

    # Step 3: optional visual review loop
    last_good_tex = tex
    if args.visual_review and png_path is not None:
        for i in range(1, args.visual_iterations + 1):
            print(f"\n[review] Visual review iteration {i}/{args.visual_iterations} ...")
            needs_fix, out = visual_review(prompt, tex, png_path, args.agent, tracker)
            if not needs_fix:
                print("[review] LGTM, figure matches description.")
                (outdir / f"review_iter_{i}.txt").write_text(out, encoding="utf-8")
                break
            print("[review] Revision proposed, recompiling ...")
            (outdir / f"review_iter_{i}.tex").write_text(out, encoding="utf-8")
            candidate = out
            tex_path.write_text(candidate, encoding="utf-8")
            ok, log, pdf = compile_latex(tex_path)
            if not ok:
                print("[review] Revised version failed to compile, keeping previous tex.")
                (outdir / f"review_iter_{i}_compile_log.txt").write_text(log, encoding="utf-8")
                tex_path.write_text(last_good_tex, encoding="utf-8")
                compile_latex(tex_path)  # regenerate the previous PDF
                break
            tex = candidate
            last_good_tex = candidate
            pdf_path = pdf
            pdf_to_png(pdf_path, png_path)
    elif args.visual_review:
        print("[review] Skipped (PNG preview not available).")

    print(f"\n[done] Figure tex: {tex_path}")
    if pdf_path is not None:
        print(f"[done] Figure pdf: {pdf_path}")
    if png_path and png_path.exists():
        print(f"[done] Figure png: {png_path}")

    _write_metadata(
        outdir, args, prompt_path, tracker,
        compiled=True,
        visual_reviewed=args.visual_review and png_path is not None)
    print(tracker.summary())


def _write_metadata(
    outdir: Path,
    args: argparse.Namespace,
    prompt_path: Path,
    tracker: CostTracker,
    compiled: bool,
    visual_reviewed: bool) -> None:
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "prompt_file": str(prompt_path),
        "colors_file": args.colors,
        "agent": args.agent,
        "chat_model": CHAT_MODEL,
        "reasoning_model": REASONING_MODEL,
        "max_compile_retries": args.max_compile_retries,
        "visual_review": args.visual_review,
        "visual_iterations": args.visual_iterations,
        "compiled": compiled,
        "visual_reviewed": visual_reviewed,
        "cost": {
            "total_usd": round(tracker.total, 6),
            "entries": tracker.entries,
        },
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate publication-quality figures by having an LLM write TikZ/LaTeX.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--prompt", required=True,
        help="Path to the figure description .txt file")
    p.add_argument(
        "--outdir",
        default=str(SCRIPT_DIR / "outputs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")),
        help="Output directory for .tex.pdf, logs, metadata")
    p.add_argument(
        "--colors", default=None,
        help="Optional path to a color schema JSON to inject into the prompt")
    p.add_argument(
        "--agent", choices=["gpt", "claude"], default="claude",
        help="Which LLM to use (default: claude, better at TikZ)")
    p.add_argument(
        "--max-compile-retries", type=int, default=3,
        help="Max LLM-driven compile-fix attempts after the first try (default: 3)")
    p.add_argument(
        "--visual-review", action="store_true",
        help="After a successful compile, render the PDF to PNG and ask the LLM "
             "to critique the rendering. Requires pymupdf (pip install pymupdf).")
    p.add_argument(
        "--visual-iterations", type=int, default=2,
        help="Max visual-review iterations when --visual-review is set (default: 2)")
    p.add_argument(
        "--png", action="store_true",
        help="Always produce a PNG preview of the final figure (requires pymupdf)")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not API_KEY:
        print("ERROR: AZURE_OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)
    if args.agent == "claude" and not ANTHROPIC_BASE:
        print("ERROR: REASONING_ENDPOINT not set but --agent claude selected.")
        sys.exit(1)
    if args.agent == "gpt" and not CHAT_BASE:
        print("ERROR: CHAT_ENDPOINT not set but --agent gpt selected.")
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()
