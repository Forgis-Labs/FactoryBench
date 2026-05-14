"""``factorybench`` CLI."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import click

from . import __version__
from .data import REPO_ID, load_split
from .evaluate import L4_REQUIRES_JUDGES_MSG, evaluate
from .parse import ParseError, parse_output
from .prompt import render_prompt
from .registry import list_models
from .result import ItemResult, Result
from .score import chance_of, score_item
from .types import AnswerFormat


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

PROVIDER_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "DeepSeek": "DEEPSEEK_API_KEY",
}


def _import_user_module(spec: str) -> None:
    """Import a user module so its ``@register_model`` decorators fire.

    Accepts either a dotted module path (``my_pkg.models``) or a filesystem
    path to a Python file.
    """
    p = Path(spec)
    if p.exists() and p.suffix == ".py":
        mod_name = f"_factorybench_user_{p.stem}"
        spec_obj = importlib.util.spec_from_file_location(mod_name, p)
        if spec_obj is None or spec_obj.loader is None:
            raise click.ClickException(f"could not load module from {p}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[mod_name] = module
        spec_obj.loader.exec_module(module)
        return
    importlib.import_module(spec)


def _filter_template(items, template: str | None):
    if not template:
        return items
    if "." not in template:
        raise click.BadParameter("template must look like 'L2.7' (level.template_id)")
    lvl_str, tid_str = template.split(".", 1)
    lvl = int(lvl_str.lstrip("Ll"))
    tid = int(tid_str)
    return [it for it in items if it.level == lvl and it.template_id == tid]


def _default_output_path(model: str, level: str, split: str) -> Path:
    safe_model = model.replace("/", "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("./results") / f"{safe_model}_{level}_{split}_{ts}.json"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="factorybench")
def cli():
    """FactoryBench: evaluate models on the FactoryBench test split."""


# -- info -------------------------------------------------------------------- #

@cli.command("info")
def cmd_info():
    """Show install + provider configuration."""
    click.echo(f"factorybench version : {__version__}")
    click.echo(f"dataset repo         : {REPO_ID} (HF Hub)")
    cache = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE") or "~/.cache/huggingface"
    click.echo(f"hf cache             : {cache}")
    click.echo("")
    click.echo("provider credentials:")
    for name, env in PROVIDER_KEYS.items():
        status = "configured" if os.environ.get(env) else "missing"
        click.echo(f"  {name:<10s} ({env:<20s}): {status}")
    click.echo("")
    click.echo("registered models:")
    for m in list_models():
        click.echo(f"  {m}")
    if not list_models():
        click.echo("  (none -- use @factorybench.register_model in a script)")


# -- list -------------------------------------------------------------------- #

@cli.group("list")
def cmd_list():
    """List available resources."""


@cmd_list.command("models")
def list_models_cmd():
    """List registered + built-in providers."""
    click.echo("Registered (custom):")
    for m in list_models():
        click.echo(f"  {m}")
    if not list_models():
        click.echo("  (none)")
    click.echo("\nBuilt-in providers (string patterns):")
    click.echo("  mock           : canned-response model, no API calls")
    click.echo("  gpt-*          : OpenAI chat completions    (OPENAI_API_KEY)")
    click.echo("  claude-*       : Anthropic Messages API     (ANTHROPIC_API_KEY)")
    click.echo("  deepseek-*     : DeepSeek (OpenAI-compatible) (DEEPSEEK_API_KEY)")


@cmd_list.command("levels")
@click.option("--split", default="test", show_default=True)
def list_levels_cmd(split: str):
    """Show item counts per level."""
    for lvl in (1, 2, 3, 4):
        items = load_split(level=lvl, split=split)
        click.echo(f"  L{lvl} : {len(items):>6d} items")


@cmd_list.command("templates")
@click.option("--level", required=True, help="e.g. L2 or 2")
@click.option("--split", default="test", show_default=True)
def list_templates_cmd(level: str, split: str):
    """Show templates available within a level."""
    items = load_split(level=level, split=split)
    by_tid: dict[int, list] = {}
    for it in items:
        by_tid.setdefault(it.template_id, []).append(it)
    for tid in sorted(by_tid):
        rows = by_tid[tid]
        ttype = rows[0].template_type
        fmt_counts = Counter(r.answer_format.value for r in rows)
        fmt_str = ", ".join(f"{k}={v}" for k, v in fmt_counts.items())
        click.echo(f"  L{rows[0].level}.{tid}  type={ttype:<28s} n={len(rows):<5d} {fmt_str}")


# -- evaluate ---------------------------------------------------------------- #

@cli.command("evaluate")
@click.option("--model", required=True, help="Registered name, built-in spec (gpt-5.1, claude-sonnet-4.6, mock), or import path.")
@click.option("--level", default="all", show_default=True, help="L1 | L2 | L3 | L4 | all  (L4 requires --judges)")
@click.option("--split", default="test", show_default=True, help="test | validation | train | mini")
@click.option("--max-items", type=int, default=None, help="Cap items per level.")
@click.option("--template", default=None, help="Filter to a single template, e.g. L2.7")
@click.option("--dataset-version", default=None, help="HF dataset revision / tag.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Result JSON path (default ./results/...).")
@click.option("--import", "import_spec", default=None, help="Pre-import a module / .py file so @register_model decorators fire.")
@click.option("--judges", default=None, help="L4 judge ensemble. 'paper-default' or a comma-separated list of judge specs.")
@click.option("--judge-cache-only", is_flag=True, help="Refuse new judge API calls; rely on the on-disk cache.")
@click.option("--judge-concurrency", type=int, default=None, help="Parallel judges per item. Default: min(#judges, 8). Set 1 for sequential.")
@click.option("--concurrency", type=int, default=1, show_default=True, help="Parallel candidate-model calls (thread pool).")
@click.option("--resume", "resume_path", type=click.Path(path_type=Path, exists=True), default=None, help="Resume from a prior Result JSON; items already scored are reused.")
@click.option("--dry-run", is_flag=True, help="Print a cost preview and exit without calling any model.")
@click.option("-y", "--yes", "assume_yes", is_flag=True, help="Skip the cost confirmation prompt (CI / scripted runs).")
@click.option("--no-progress", is_flag=True, help="Disable the per-item progress bar.")
def cmd_evaluate(model, level, split, max_items, template, dataset_version, output, import_spec, judges, judge_cache_only, judge_concurrency, concurrency, resume_path, dry_run, assume_yes, no_progress):
    """Run a model over the test split and write a Result JSON."""
    if import_spec:
        _import_user_module(import_spec)

    panel = _resolve_panel_or_die(judges, cache_only=judge_cache_only)
    if panel is not None and judge_concurrency is not None:
        if judge_concurrency < 1:
            raise click.BadParameter("--judge-concurrency must be >= 1")
        panel.concurrency = judge_concurrency
    if concurrency < 1:
        raise click.BadParameter("--concurrency must be >= 1")

    # Cost preview / gating. Skip when only re-aggregating from cache (no spend).
    estimated_cost: float | None = None
    if not judge_cache_only:
        proceed, estimated_cost = _cost_gate(
            model, level, split, max_items, panel,
            dry_run=dry_run, assume_yes=assume_yes,
        )
        if not proceed:
            return

    try:
        return _do_evaluate(
            model, level, split, max_items, template, dataset_version, output,
            no_progress, panel, concurrency=concurrency, resume_path=resume_path,
            estimated_cost=estimated_cost,
        )
    except (KeyError, NotImplementedError, ImportError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _cost_gate(model, level, split, max_items, panel, *, dry_run, assume_yes) -> tuple[bool, float | None]:
    """Print a cost preview; possibly prompt for confirmation.

    Returns ``(should_proceed, estimated_total_cost)``. ``estimated_total_cost`` is
    ``None`` when estimation failed.
    """
    from .cost import estimate_cost

    try:
        est = estimate_cost(model=model, level=level, split=split, max_items=max_items, judges=panel)
    except Exception as exc:
        # Cost estimation must never block a run on its own; if it fails, warn and proceed.
        click.echo(f"warning: cost estimate failed ({type(exc).__name__}: {exc}); skipping preview", err=True)
        return True, None

    if dry_run:
        click.echo(est.format())
        click.echo("\n--dry-run: not calling any model. Exiting.")
        return False, est.total_cost

    needs_prompt = (est.n_l4_items > 0) or (est.total_cost >= 1.0)
    if not needs_prompt or assume_yes:
        return True, est.total_cost

    click.echo(est.format())
    if not click.confirm("\nContinue?", default=False):
        click.echo("aborted.")
        return False, est.total_cost
    return True, est.total_cost


def _do_evaluate(model, level, split, max_items, template, dataset_version, output, no_progress, panel, *, concurrency=1, resume_path=None, estimated_cost=None):
    if template:
        # We need to filter post-load; let evaluate() do its own load + run.
        items = load_split(
            level=level if not template else _level_from_template(template),
            split=split,
            revision=dataset_version,
            max_items=max_items,
        )
        items = _filter_template(items, template)
        if not items:
            raise click.ClickException(f"no items match template {template}")
        # Gate L4 templates through the panel.
        if items[0].level == 4 and panel is None:
            raise click.ClickException(L4_REQUIRES_JUDGES_MSG)
        from .evaluate import _run_predict, _score_one, _adapter_model_id, _summarize_usage
        from .cost import compute_cost_from_usage
        from .registry import get_model
        instance = get_model(model)
        prompts = [render_prompt(it) for it in items]
        import time
        from datetime import timedelta
        t0 = time.perf_counter()
        raws, candidate_usages = _run_predict(instance, prompts, progress=not no_progress, concurrency=concurrency)
        wall = timedelta(seconds=time.perf_counter() - t0)
        tokens_used = _summarize_usage(
            candidate_model_id=_adapter_model_id(instance, fallback=model),
            candidate_usages=candidate_usages,
            panel=panel,
        )
        result = Result(
            model_name=model,
            items=[_score_one(it, raw, panel=panel) for it, raw in zip(items, raws)],
            wall_time=wall,
            judges=list(panel.judge_specs) if panel else [],
            tokens_used=tokens_used,
            cost=compute_cost_from_usage(tokens_used),
        )
    else:
        result = evaluate(
            model=model,
            level=level,
            split=split,
            revision=dataset_version,
            max_items=max_items,
            progress=not no_progress,
            judges=panel,
            concurrency=concurrency,
            resume_from=resume_path,
        )

    out_path = output or _default_output_path(model, level, split)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)

    click.echo("")
    click.echo(f"  model               : {result.model_name}")
    click.echo(f"  items               : {len(result.items)}")
    click.echo(f"  score (chance-corr) : {_fmt(result.score)}")
    click.echo(f"  raw accuracy        : {_fmt(result.raw_accuracy)}")
    click.echo(f"  parse failures      : {len(result.parse_failures())}")
    click.echo(f"  wall time           : {result.wall_time}")
    if result.judges:
        click.echo(f"  judges              : {', '.join(result.judges)}")
        click.echo(f"  judge mode          : {result.judge_mode()}")
        click.echo(f"  fleiss kappa (L4)   : {_fmt(result.fleiss_kappa())}")
    if _has_real_usage(result.tokens_used):
        actual_str = f"${result.cost:,.4f}"
        if estimated_cost is not None:
            actual_str += f"  (estimate was ${estimated_cost:,.4f})"
        click.echo(f"  actual cost         : {actual_str}")
    click.echo("")
    click.echo("  by_level:")
    for k, v in result.by_level().items():
        click.echo(f"    {k:<6s}: {_fmt(v)}")
    click.echo("")
    click.echo(f"saved -> {out_path}")


def _has_real_usage(tokens_used: dict | None) -> bool:
    if not tokens_used:
        return False
    cand = (tokens_used.get("candidate") or {})
    if (cand.get("calls") or 0) > 0:
        return True
    for j in (tokens_used.get("judges") or {}).values():
        if (j.get("calls") or 0) > 0:
            return True
    return False


def _resolve_panel_or_die(judges_flag, *, cache_only: bool = False):
    """Translate ``--judges`` into a JudgePanel, after checking API keys."""
    if judges_flag is None:
        if cache_only:
            raise click.ClickException("--judge-cache-only requires --judges to be set too")
        return None
    from .judges import parse_judges_flag, precheck_credentials
    try:
        panel = parse_judges_flag(judges_flag)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    panel.cache_only = cache_only
    # In cache-only mode, we never call provider APIs, so skip the key precheck.
    if cache_only:
        if panel.is_single_judge and not panel.is_paper_default:
            click.echo(
                f"note: single-judge mode ({panel.judge_specs[0]}) -- results will be "
                "flagged 'single-judge' and are not directly comparable to the paper.",
                err=True,
            )
        return panel
    missing = precheck_credentials(panel)
    if missing:
        lines = [
            f"L4 evaluation requested with --judges {judges_flag!r}, "
            "but the following credentials are missing:",
            *missing,
            "",
            "Set the missing variables, or choose a smaller ensemble with --judges.",
        ]
        raise click.ClickException("\n".join(lines))
    if panel.is_single_judge and not panel.is_paper_default:
        click.echo(
            f"note: single-judge mode ({panel.judge_specs[0]}) -- results will be "
            "flagged 'single-judge' and are not directly comparable to the paper.",
            err=True,
        )
    return panel


def _level_from_template(template: str) -> str:
    return f"L{int(template.split('.')[0].lstrip('Ll'))}"


# -- export ------------------------------------------------------------------ #

@cli.command("export")
@click.option("--level", default="all", show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--max-items", type=int, default=None)
@click.option("--template", default=None)
@click.option("--dataset-version", default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True, help="JSONL path.")
def cmd_export(level, split, max_items, template, dataset_version, output):
    """Export rendered prompts as JSONL (for offline inference)."""
    items = load_split(level=level, split=split, revision=dataset_version, max_items=max_items)
    items = _filter_template(items, template)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for it in items:
            row = {
                "item_id": it.id,
                "level": it.level,
                "template_id": it.template_id,
                "answer_format": it.answer_format.value,
                "prompt": render_prompt(it),
            }
            f.write(json.dumps(row) + "\n")
    click.echo(f"wrote {len(items)} rows -> {output}")


# -- score ------------------------------------------------------------------- #

@cli.command("score")
@click.option("--predictions", type=click.Path(path_type=Path, exists=True), required=True, help="JSONL with {item_id, prediction}.")
@click.option("--level", default="all", show_default=True, help="Which level(s) the predictions cover.")
@click.option("--split", default="test", show_default=True)
@click.option("--dataset-version", default=None)
@click.option("--output", type=click.Path(path_type=Path), required=True, help="Result JSON path.")
@click.option("--model-name", default="offline", show_default=True)
@click.option("--judges", default=None, help="L4 judge ensemble. 'paper-default' or a comma-separated list.")
@click.option("--judge-cache-only", is_flag=True, help="Refuse new judge API calls; rely on the on-disk cache.")
@click.option("--judge-concurrency", type=int, default=None, help="Parallel judges per item. Default: min(#judges, 8). Set 1 for sequential.")
@click.option("--import", "import_spec", default=None, help="Pre-import a module / .py file so @register_model decorators fire (useful for custom judges).")
def cmd_score(predictions, level, split, dataset_version, output, model_name, judges, judge_cache_only, judge_concurrency, import_spec):
    """Score a predictions JSONL produced by an external runner."""
    if import_spec:
        _import_user_module(import_spec)
    panel = _resolve_panel_or_die(judges, cache_only=judge_cache_only)
    if panel is not None and judge_concurrency is not None:
        if judge_concurrency < 1:
            raise click.BadParameter("--judge-concurrency must be >= 1")
        panel.concurrency = judge_concurrency

    preds: dict[str, str] = {}
    with predictions.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            preds[obj["item_id"]] = obj.get("prediction") or ""

    items = load_split(level=level, split=split, revision=dataset_version)
    by_id = {it.id: it for it in items}

    missing = set(preds) - set(by_id)
    if missing:
        click.echo(f"warning: {len(missing)} prediction(s) reference unknown item ids", err=True)

    # If any of the predicted items are L4 and no judges provided, fail loudly.
    l4_present = any(by_id[i].level == 4 for i in preds if i in by_id)
    if l4_present and panel is None:
        raise click.ClickException(L4_REQUIRES_JUDGES_MSG)

    from .evaluate import _score_one

    item_results: list[ItemResult] = []
    for item_id, item in by_id.items():
        raw = preds.get(item_id)
        if raw is None:
            continue  # not scored -- item missing from predictions
        item_results.append(_score_one(item, raw, panel=panel))

    result = Result(
        model_name=model_name,
        items=item_results,
        judges=list(panel.judge_specs) if panel else [],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)

    click.echo(f"scored {len(item_results)} items -> {output}")
    click.echo(f"  score (chance-corr) : {_fmt(result.score)}")
    click.echo(f"  raw accuracy        : {_fmt(result.raw_accuracy)}")
    click.echo(f"  parse failures      : {len(result.parse_failures())}")
    if result.judges:
        click.echo(f"  judges              : {', '.join(result.judges)}")
        click.echo(f"  judge mode          : {result.judge_mode()}")
        click.echo(f"  fleiss kappa (L4)   : {_fmt(result.fleiss_kappa())}")


# -- cost -------------------------------------------------------------------- #

@cli.command("cost")
@click.option("--model", required=True, help="Model spec for the candidate (gpt-5.1, claude-sonnet-4.6, mock, ...).")
@click.option("--level", default="all", show_default=True, help="L1 | L2 | L3 | L4 | all")
@click.option("--split", default="test", show_default=True)
@click.option("--max-items", type=int, default=None, help="Cap items per level.")
@click.option("--judges", default=None, help="L4 judge ensemble. 'paper-default' or a comma-separated list.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"], case_sensitive=False), default="text", show_default=True)
def cmd_cost(model, level, split, max_items, judges, fmt):
    """Estimate cost (and wall time) of an evaluate call. No API calls made."""
    from .cost import estimate_cost

    panel = None
    if judges is not None:
        from .judges import parse_judges_flag
        try:
            panel = parse_judges_flag(judges)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        # Cost preview only -- skip credential precheck so it works offline.

    try:
        est = estimate_cost(model=model, level=level, split=split, max_items=max_items, judges=panel)
    except (KeyError, NotImplementedError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if fmt == "json":
        click.echo(json.dumps(est.to_dict(), indent=2))
    else:
        click.echo(est.format())


# -- compare ----------------------------------------------------------------- #

@cli.command("compare")
@click.argument("runs", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["markdown", "latex", "json"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format.",
)
@click.option("--decimals", type=int, default=4, show_default=True, help="Cells are formatted with this many decimal places.")
@click.option("--no-bold", is_flag=True, help="Disable bolding the best score in each row.")
@click.option("--name", "names", multiple=True, help="Override model column name. Pass once per RUNS argument in the same order.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Write to a file instead of stdout.")
def cmd_compare(runs, fmt, decimals, no_bold, names, output):
    """Render a model x level comparison from saved Result JSON files."""
    from .compare import compare as _compare

    if names and len(names) != len(runs):
        raise click.BadParameter(
            f"got {len(names)} --name values for {len(runs)} runs; pass one --name per run or none at all"
        )

    if names:
        results = {label: path for label, path in zip(names, runs)}
    else:
        # Default to the file stem; on collision, fall back to the saved model_name.
        results = {}
        for path in runs:
            label = path.stem
            if label in results:
                label = f"{label}@{len(results)}"
            results[label] = path

    comp = _compare(results)

    if fmt == "markdown":
        text = comp.to_markdown(decimals=decimals, bold_best=not no_bold)
    elif fmt == "latex":
        text = comp.to_latex(decimals=decimals, bold_best=not no_bold)
    else:
        text = comp.to_json(decimals=decimals)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        click.echo(f"wrote -> {output}")
    else:
        click.echo(text)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def _fmt(x: float) -> str:
    import math
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.4f}"


if __name__ == "__main__":
    cli()
