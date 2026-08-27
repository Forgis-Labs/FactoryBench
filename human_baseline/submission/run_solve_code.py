"""
Executes every item's `solve_code` in sampled_dataset.json and stores what it printed
back into that item's `solve_output` field.

Why this exists: `build_notebook.py` is a pure renderer -- sampled_dataset.json in,
notebook out, no data access. So the notebook's code-cell outputs have to be captured
here, once, against the real raw data, and carried in the JSON alongside the code.

A non-zero exit or an AssertionError from any item's solve_code is a hard failure: each
solve_code asserts its own derived answer against the item's stored `answer` (level 3)
or `acceptance_bounds` (level 1), so this script doubles as the dataset's regression
test. Run it after any change to solve_code, then re-run build_notebook.py.

Usage:
  python3 run_solve_code.py            # run all items, write solve_output back
  python3 run_solve_code.py --check    # run all items, verify only, write nothing
"""
import json
import subprocess
import sys
import tempfile
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLED = os.path.join(HERE, "sampled_dataset.json")


def run_one(code):
    """Runs one solve_code in a fresh interpreter. Returns (ok, combined_output)."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="solve_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(code)
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True,
                              timeout=900)
        out = proc.stdout
        if proc.returncode != 0:
            out += "\n--- STDERR ---\n" + proc.stderr
        return proc.returncode == 0, out
    finally:
        os.unlink(path)


def main():
    check_only = "--check" in sys.argv
    items = json.load(open(SAMPLED))
    failed = []
    for i, it in enumerate(items, 1):
        qid = it["original_question_id"]
        code = it.get("solve_code")
        if not code:
            print(f"[{i}/{len(items)}] {qid}  SKIP -- no solve_code")
            continue
        ok, out = run_one(code)
        it["solve_output"] = out
        tag = "ok" if ok else "FAILED"
        print(f"[{i}/{len(items)}] L{it['level']} t{it['template_id']} {qid}  {tag} "
              f"({len(out.splitlines())} lines of output)")
        if not ok:
            failed.append(qid)
            print(out)

    if not check_only:
        json.dump(items, open(SAMPLED, "w"), indent=1)
        print(f"\nWrote solve_output for {len(items)} items to {SAMPLED}")

    print("RESULT:", "all solve_code ran clean" if not failed
          else f"{len(failed)} item(s) FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
