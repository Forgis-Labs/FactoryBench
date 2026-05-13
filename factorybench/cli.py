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
@click.option("--no-progress", is_flag=True, help="Disable the per-item progress bar.")
def cmd_evaluate(model, level, split, max_items, template, dataset_version, output, import_spec, judges, no_progress):
    """Run a model over the test split and write a Result JSON."""
    if import_spec:
        _import_user_module(import_spec)

    panel = _resolve_panel_or_die(judges)

    try:
        return _do_evaluate(model, level, split, max_items, template, dataset_version, output, no_progress, panel)
    except (KeyError, NotImplementedError, ImportError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _do_evaluate(model, level, split, max_items, template, dataset_version, output, no_progress, panel):
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
        from .evaluate import _run_predict, _score_one
        from .registry import get_model
        instance = get_model(model)
        prompts = [render_prompt(it) for it in items]
        import time
        from datetime import timedelta
        t0 = time.perf_counter()
        raws = _run_predict(instance, prompts, progress=not no_progress)
        wall = timedelta(seconds=time.perf_counter() - t0)
        result = Result(
            model_name=model,
            items=[_score_one(it, raw, panel=panel) for it, raw in zip(items, raws)],
            wall_time=wall,
            judges=list(panel.judge_specs) if panel else [],
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
    click.echo("")
    click.echo("  by_level:")
    for k, v in result.by_level().items():
        click.echo(f"    {k:<6s}: {_fmt(v)}")
    click.echo("")
    click.echo(f"saved -> {out_path}")


def _resolve_panel_or_die(judges_flag):
    """Translate ``--judges`` into a JudgePanel, after checking API keys."""
    if judges_flag is None:
        return None
    from .judges import parse_judges_flag, precheck_credentials
    try:
        panel = parse_judges_flag(judges_flag)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
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
@click.option("--import", "import_spec", default=None, help="Pre-import a module / .py file so @register_model decorators fire (useful for custom judges).")
def cmd_score(predictions, level, split, dataset_version, output, model_name, judges, import_spec):
    """Score a predictions JSONL produced by an external runner."""
    if import_spec:
        _import_user_module(import_spec)
    panel = _resolve_panel_or_die(judges)

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
