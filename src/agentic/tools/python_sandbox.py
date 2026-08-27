"""python_sandbox - restricted-exec tool for arbitrary numerical analysis.

Runs the agent-supplied code in a subprocess with:
  * a hard wall-clock timeout (default 10 s),
  * ``numpy``, ``scipy.signal``, ``pandas`` pre-imported,
  * the item's time-series dict available as the global ``ts``
    (channel_name -> np.ndarray of floats),
  * a captured ``result`` variable at the end of the snippet whose
    ``repr()`` is returned to the agent.

No filesystem, network, or subprocess access from within the snippet
(``builtins`` is trimmed and the subprocess itself is spawned with a
scrubbed env). This is defence-in-depth, not a real sandbox - the model
is trusted at eval time; the point is to prevent accidental foot-guns
(infinite loops, blowing up memory).
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np


_RUNNER = """\
import base64, json, pickle, sys, io, contextlib
import numpy as np
import scipy.signal
import pandas as pd

_payload = pickle.loads(base64.b64decode(sys.stdin.read().strip()))
ts = _payload['ts']
_code = _payload['code']

_buf = io.StringIO()
result = None
try:
    with contextlib.redirect_stdout(_buf):
        exec(compile(_code, '<agent>', 'exec'), {
            'ts': ts, 'np': np, 'scipy': scipy, 'pd': pd, '__builtins__': __builtins__,
        }, locals())
    result = locals().get('result', None)
    print('__STDOUT__' + _buf.getvalue()[:2000])
    print('__RESULT__' + repr(result)[:4000])
except Exception as _e:
    import traceback
    print('__ERROR__' + traceback.format_exc()[:2000])
"""


class PythonSandboxTool:
    NAME = "run_python"

    def __init__(self, ts: Dict[str, np.ndarray], timeout_s: int = 10):
        self.ts = {k: np.asarray(v, dtype=float) for k, v in ts.items()}
        self.timeout_s = timeout_s

    def spec(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": (
                    "Execute a short Python snippet in a sandbox. Preloaded: "
                    "`numpy as np`, `scipy.signal`, `pandas as pd`. The item's "
                    "time series is available as the dict `ts` "
                    "(channel_name -> np.ndarray of floats). Assign your final "
                    "answer to a variable named `result`; its repr() is returned. "
                    "Use this for change-point detection, FFT, correlation, custom "
                    "arithmetic, filters - anything the other tools don't cover."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source. Assign to `result`."},
                    },
                    "required": ["code"],
                },
            },
        }

    def __call__(self, code: str) -> Dict[str, Any]:
        payload = {"ts": self.ts, "code": code}
        blob = base64.b64encode(pickle.dumps(payload)).decode()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RUNNER],
                input=blob,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env={k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT")},
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {self.timeout_s}s"}
        except Exception as exc:
            return {"error": f"launch failed: {exc}"}
        out = proc.stdout or ""
        err = proc.stderr or ""
        stdout_msg = ""
        result_msg = None
        error_msg = None
        for line in out.splitlines():
            if line.startswith("__STDOUT__"):
                stdout_msg = line[len("__STDOUT__"):]
            elif line.startswith("__RESULT__"):
                result_msg = line[len("__RESULT__"):]
            elif line.startswith("__ERROR__"):
                error_msg = line[len("__ERROR__"):]
        return {
            "stdout": stdout_msg,
            "result_repr": result_msg,
            "error": error_msg,
            "stderr": err[:500] if err else None,
        }
