# Contributing

Thanks for your interest in `factorybench`. Bug reports, fixes, new provider
adapters, and rubric improvements are all welcome.

## Dev setup

```bash
git clone https://github.com/Forgis-Labs/FactoryBench
cd factorybench
pip install -e ".[all,dev]"
pytest tests/ -q
```

Python 3.10+ is required.

## Running the suite

- `pytest tests/ -q` -- full unit suite (no network, ~1s).
- `pytest tests/test_<name>.py -v` -- focus on one module.
- `factorybench info` -- sanity-check your install + provider credentials.

## What a good PR looks like

- One focused change per PR. Big refactors are easier to review when split.
- Tests for new behavior. The bar isn't 100% coverage, but new public API
  surface or a non-trivial bug fix should come with at least one test.
- A `CHANGELOG.md` entry under the next-version section (one bullet, terse,
  matches the existing voice).
- If you touched the CLI surface or the Python public API, run through the
  relevant section of the README and update it where it drifts.

## Areas where help is especially welcome

- New built-in provider adapters (vLLM-served local models, Together,
  Fireworks, Bedrock, ...). The contract is `predict(prompt) -> str` plus an
  optional `predict_with_usage(prompt) -> (str, usage_dict | None)`. See
  `factorybench/adapters/` for the existing OpenAI / Anthropic adapters.
- L4 rubric experiments. The current rubric is a 0 / 0.5 / 1 scale; we'd be
  interested in alternative rubrics with per-criterion sub-scores.
- Performance: batched / async candidate-model paths beyond the current
  thread-pool approach.

## What we will probably not merge

- Auto-loading `.env` files (deliberately matches OpenAI / Anthropic SDK
  behavior; see README "Provider credentials").
- New optional dependencies for things that already work with a heuristic
  fallback, unless the gain is clearly worth the install-size cost.
- Wholesale changes to the rubric without empirical justification --
  reproducibility of paper numbers is a constraint.

## Cutting a release

See [RELEASING.md](RELEASING.md). Short version: bump `pyproject.toml` +
`factorybench/__init__.py` + `CHANGELOG.md`, commit, `git tag -a v0.0.X`,
push tag. The `release.yml` workflow does the rest via PyPI Trusted Publishing.

## Reporting bugs

Please include the output of `factorybench info --json` in any bug report.
That captures the package version, dataset revision, Python version, and
which providers are configured -- which is what we ask for first anyway.
