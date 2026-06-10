# Releasing `factorybench`

Releases publish to [PyPI](https://pypi.org/project/factorybench/) via GitHub
Actions and **Trusted Publishing** (OIDC) -- no PyPI API tokens are stored
anywhere. The workflow is `.github/workflows/release.yml`. It fires on any
pushed tag matching `v*`.

## One-time setup (PyPI side)

Done **once per project**, by a maintainer with PyPI access:

1. Create the PyPI account: <https://pypi.org/account/register/>
2. Reserve the project name with a "pending publisher" so the first release
   can claim it without an existing release.
   - Go to <https://pypi.org/manage/account/publishing/>.
   - Click **Add a new pending publisher**.
   - Fill in:
     - PyPI Project Name: `factorybench`
     - Owner: `Forgis-Labs`
     - Repository name: `FactoryBench`
     - Workflow name: `release.yml`
     - Environment name: `pypi`
3. (Optional but recommended) Enable 2FA on the PyPI account.

After the first successful release the pending publisher becomes a permanent
trusted publisher, and subsequent releases don't require any additional setup.

## One-time setup (GitHub side)

1. Create a GitHub environment named `pypi` on the repo:
   - **Settings -> Environments -> New environment -> `pypi`**.
   - Optional: add a "required reviewer" so releases need manual approval, or
     restrict to `main` only.
2. Confirm Actions can run with `id-token: write` (default for public repos).

## Cutting a release

```bash
# 1. Bump the version in pyproject.toml AND factorybench/__init__.py.
#    The release workflow refuses to publish if these disagree with the tag.
#    Add a CHANGELOG.md entry for the new version.

# 2. Commit the bump.
git add pyproject.toml factorybench/__init__.py CHANGELOG.md
git commit -m "release v0.0.X"

# 3. Tag + push.
git tag -a v0.0.X -m "v0.0.X"
git push origin main --tags
```

That's it. Pushing the tag triggers `release.yml`, which:

1. Verifies the tag (`v0.0.X`) matches `pyproject.toml`'s `version`.
2. Builds the sdist + wheel with `python -m build`.
3. Reinstalls the built wheel into a clean venv on Python 3.10 / 3.11 / 3.12 /
   3.13 and runs the test suite against the installed package (not the source
   tree). Catches "works on dev machine, broken in wheel" regressions.
4. Uploads to PyPI via Trusted Publishing.

You can follow progress at
<https://github.com/Forgis-Labs/FactoryBench/actions/workflows/release.yml>.

## Verifying a release

```bash
pip install --upgrade factorybench
factorybench --version    # should print v0.0.X
factorybench info         # should not crash; tiktoken, hf_hub etc. all reachable
```

## Rolling back

PyPI does **not** allow re-uploading a file under the same version (even if
yanked). If a release is broken:

1. Yank the version on PyPI (UI: project page -> Manage -> Releases -> Yank).
   Yanked versions stay installable for users who pin them, but new
   `pip install factorybench` calls skip them.
2. Bump the version (e.g., `0.0.X+1`) and re-release.

## TestPyPI rehearsals (optional)

For risky releases (large refactors, dependency changes), test against
TestPyPI first:

1. Add a second "pending publisher" entry on <https://test.pypi.org/> with
   the same settings, environment name `testpypi`.
2. Run `release.yml` manually against a `0.0.X-rcN` tag, after temporarily
   pointing the publish step at TestPyPI (`repository-url:
   https://test.pypi.org/legacy/`).
3. Verify with `pip install -i https://test.pypi.org/simple/ factorybench`.

We don't do this for every release; it's the escape hatch when changes are
non-trivial.
