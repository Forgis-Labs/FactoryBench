import json

from factorybench.compare import Comparison, compare
from factorybench.result import ItemResult, Result


def _result(name, *scores_by_level):
    """scores_by_level: tuples of (level, [scores])."""
    items = []
    for level, scores in scores_by_level:
        for s in scores:
            items.append(ItemResult(
                id=f"{name}_{level}_{len(items)}",
                level=level,
                template_id=1,
                template_type="t",
                answer_format="scalar_range",
                dataset="aursad",
                fault_id=0,
                raw_output="0",
                parsed=0,
                score=s,
                chance=0.0,
            ))
    return Result(model_name=name, items=items)


def test_compare_requires_inputs():
    import pytest
    with pytest.raises(ValueError):
        compare({})


def test_compare_dict_results():
    a = _result("a", (1, [1.0, 0.0]), (2, [0.5]))
    b = _result("b", (1, [0.5, 0.5]), (2, [1.0]))
    comp = compare({"a": a, "b": b})
    grid = comp.score_grid()
    assert list(grid.columns) == ["a", "b"]
    assert grid.loc["L1", "a"] == 0.5
    assert grid.loc["L1", "b"] == 0.5
    assert grid.loc["L2", "a"] == 0.5
    assert grid.loc["L2", "b"] == 1.0


def test_compare_absent_levels_are_nan():
    a = _result("a", (1, [1.0]))  # only L1
    b = _result("b", (1, [0.0]), (2, [1.0]))
    comp = compare({"a": a, "b": b})
    grid = comp.score_grid()
    # 'a' has no L2 -> NaN.
    import math
    assert math.isnan(grid.loc["L2", "a"])
    assert grid.loc["L2", "b"] == 1.0


def test_markdown_bolds_row_best():
    a = _result("a", (1, [1.0]))
    b = _result("b", (1, [0.0]))
    md = compare({"a": a, "b": b}).to_markdown()
    # 'a' wins -> 'a' bolded on the L1 row.
    assert "**1.0000**" in md
    assert "**0.0000**" not in md


def test_markdown_disable_bold():
    a = _result("a", (1, [1.0]))
    md = compare({"a": a}).to_markdown(bold_best=False)
    assert "1.0000" in md
    assert "**" not in md


def test_latex_booktabs_default():
    a = _result("a", (1, [1.0]))
    latex = compare({"a": a}).to_latex()
    assert r"\toprule" in latex
    assert r"\bottomrule" in latex
    assert r"\textbf{1.0000}" in latex


def test_latex_no_booktabs():
    a = _result("a", (1, [1.0]))
    latex = compare({"a": a}).to_latex(booktabs=False)
    assert r"\toprule" not in latex
    assert r"\hline" in latex


def test_to_json_round_trip():
    a = _result("a", (1, [1.0]))
    b = _result("b", (1, [0.5]))
    out = compare({"a": a, "b": b}).to_json()
    parsed = json.loads(out)
    assert parsed["models"] == ["a", "b"]
    assert parsed["grid"]["L1"]["a"] == 1.0
    assert parsed["grid"]["L1"]["b"] == 0.5
    assert parsed["parse_failures"] == {"a": 0, "b": 0}


def test_from_paths(tmp_path):
    a = _result("a", (1, [1.0]))
    p = tmp_path / "a.json"
    a.save(p)
    # Path-based form.
    comp = compare({"loaded": p})
    assert comp.score_grid().loc["L1", "loaded"] == 1.0


def test_iterable_of_paths_uses_stem(tmp_path):
    a = _result("a", (1, [1.0]))
    p = tmp_path / "my-run.json"
    a.save(p)
    comp = compare([p])
    assert list(comp.results.keys()) == ["my-run"]
