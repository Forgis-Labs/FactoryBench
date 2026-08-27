# LaTeX Figure Generator

Generates publication-quality figures by asking an LLM to write a standalone
TikZ/LaTeX document, compiling it with `latexmk`, and feeding any compilation
errors back to the model until it builds. Optionally runs a visual review
pass against the rendered PNG.

The output is a real, editable `.tex` file, not a raster image, so you get
a solid starting point that you can polish by hand or drop straight into your
paper with `\input{figure.tex}` or `\includegraphics{figure.pdf}`.

Generic: no project-specific assumptions. Pass `--prompt` for any figure and
`--colors` for an optional palette.

### Requirements

- A LaTeX distribution with `latexmk` and `pdflatex` on `PATH`
  (MiKTeX on Windows, TeX Live on Linux/macOS).
- Python dependencies:
  ```bash
  pip install requests python-dotenv openai
  pip install pymupdf   # optional, only needed for --visual-review / --png
  ```

### Setup

Create a `.env` file with your Azure AI Foundry credentials:

```
AZURE_OPENAI_API_KEY="<your_key_here>"

CHAT_MODEL="gpt-5-mini"
CHAT_ENDPOINT="<base_endpoint>/openai/v1"

REASONING_MODEL="claude-opus-4-6"
REASONING_ENDPOINT="<base_endpoint>/anthropic/v1"
```

The image endpoint is no longer needed, figures are produced as TikZ code.

### Usage

Create a text file describing the figure inside `prompts/` (or anywhere), then:

```bash
# Minimum: generate, compile, auto-fix any LaTeX errors
python scripts/neurips_image_generator.py \
    --prompt scripts/prompts/factorybench_neurips.txt

# With a project color palette injected into the prompt
python scripts/neurips_image_generator.py \
    --prompt scripts/prompts/factorybench_neurips.txt \
    --colors scripts/color_schema.json

# Use GPT instead of Claude for TikZ generation
python scripts/neurips_image_generator.py \
    --prompt scripts/prompts/fig1_overview.txt \
    --agent gpt

# After compiling, render to PNG and run a visual-review loop
python scripts/neurips_image_generator.py \
    --prompt scripts/prompts/factorybench_neurips.txt \
    --colors scripts/color_schema.json \
    --visual-review --visual-iterations 2
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | *required* | Path to the figure description `.txt` file |
| `--outdir` | `outputs/run_<timestamp>` | Output directory |
| `--colors` | *none* | Optional JSON color schema injected into the prompt |
| `--agent` | `claude` | LLM used for generation / fixing: `claude` or `gpt` |
| `--max-compile-retries` | `3` | Max LLM compile-fix attempts after the first try |
| `--visual-review` | off | Render PDF to PNG and run a visual critique loop (needs `pymupdf`) |
| `--visual-iterations` | `2` | Max iterations of the visual-review loop |
| `--png` | off | Always produce a PNG preview (needs `pymupdf`) |

### Output

Each run creates a timestamped folder in `outputs/` containing:

- `figure.tex`, final standalone LaTeX source (edit this by hand)
- `figure.pdf`, compiled PDF
- `figure.png`, PNG preview (only with `--png` or `--visual-review`)
- `figure_attempt_N.tex`, intermediate sources from the compile-fix loop
- `review_iter_N.tex` / `review_iter_N.txt`, visual-review iterations
- `compile_log.txt`, full pdflatex log from the last attempt
- `prompt.txt`, the resolved prompt (with colors injected)
- `metadata.json`, run config and cost summary
