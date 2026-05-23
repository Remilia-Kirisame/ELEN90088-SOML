"""Tests for dora_mini.summarize — pure stdlib, runs on Mac."""
import json

from dora_mini import summarize


def test_parse_run_id_valid():
    got = summarize.parse_run_id("dora_mistral7b_cs170k_r8_s42")
    assert got == {"method": "dora", "trainset": "cs170k", "r": 8, "seed": 42}


def test_parse_run_id_rejects_tier1():
    # Tier-1 run ids have no _s<seed> suffix.
    assert summarize.parse_run_id("dora_mistral7b_boolq_r8") is None
    assert summarize.parse_run_id("garbage") is None


def _run(method, trainset, r, seed, lik, gen):
    return {
        "run_id": f"{method}_mistral7b_{trainset}_r{r}_s{seed}",
        "eval_accuracy_likelihood": lik,
        "eval_accuracy_genmatch": gen,
        "eval_loss": 1.0, "train_runtime_s": 100, "peak_memory_gb": 10.0,
        "trainable_params": 1000,
    }


def test_aggregate_means_and_std():
    runs = [
        _run("lora", "boolq", 8, 42, 0.80, 0.78),
        _run("lora", "boolq", 8, 1, 0.82, 0.80),
        _run("lora", "boolq", 8, 2, 0.84, 0.82),
    ]
    cells = summarize.aggregate(runs)
    cell = cells[("boolq", "lora", 8)]
    assert cell["n"] == 3
    assert abs(cell["likelihood_mean"] - 0.82) < 1e-9
    assert abs(cell["genmatch_mean"] - 0.80) < 1e-9
    assert cell["likelihood_std"] > 0


def test_aggregate_groups_separately():
    runs = [
        _run("lora", "boolq", 4, 42, 0.80, 0.78),
        _run("dora", "boolq", 4, 42, 0.81, 0.79),
        _run("lora", "cs170k", 4, 42, 0.70, 0.40),
    ]
    cells = summarize.aggregate(runs)
    assert set(cells) == {("boolq", "lora", 4), ("boolq", "dora", 4), ("cs170k", "lora", 4)}


def _run_with_strict(method, trainset, r, seed, lik, gen, gen_strict):
    d = _run(method, trainset, r, seed, lik, gen)
    d["eval_accuracy_genmatch_strict"] = gen_strict
    return d


def test_load_strict_snapshot_present(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({
        "_meta": "test",
        "values": {"lora_mistral7b_cs170k_r8_s42": 0.5, "dora_mistral7b_cs170k_r4_s1": 0.05},
    }))
    got = summarize.load_strict_snapshot(str(path))
    assert got == {"lora_mistral7b_cs170k_r8_s42": 0.5, "dora_mistral7b_cs170k_r4_s1": 0.05}


def test_load_strict_snapshot_missing(tmp_path):
    # Absent file -> empty dict (graceful degradation; load_runs still works).
    assert summarize.load_strict_snapshot(str(tmp_path / "nonexistent.json")) == {}


def test_load_strict_snapshot_malformed(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json {{{")
    assert summarize.load_strict_snapshot(str(path)) == {}


def test_aggregate_includes_strict_when_complete():
    runs = [
        _run_with_strict("dora", "cs170k", 16, 42, 0.78, 0.77, 0.33),
        _run_with_strict("dora", "cs170k", 16, 1, 0.78, 0.80, 0.79),
        _run_with_strict("dora", "cs170k", 16, 2, 0.63, 0.78, 0.77),
    ]
    cells = summarize.aggregate(runs)
    cell = cells[("cs170k", "dora", 16)]
    assert "genmatch_strict_mean" in cell
    assert abs(cell["genmatch_strict_mean"] - (0.33 + 0.79 + 0.77) / 3) < 1e-9


def test_aggregate_omits_strict_when_partial():
    # If not every run in the cell has strict, omit it from the cell entirely (don't average partial).
    runs = [
        _run("dora", "cs170k", 16, 42, 0.78, 0.77),
        _run_with_strict("dora", "cs170k", 16, 1, 0.78, 0.80, 0.79),
        _run_with_strict("dora", "cs170k", 16, 2, 0.63, 0.78, 0.77),
    ]
    cells = summarize.aggregate(runs)
    cell = cells[("cs170k", "dora", 16)]
    assert "genmatch_strict_mean" not in cell


def test_render_includes_strict_column_when_present():
    runs = [
        _run_with_strict("dora", "cs170k", 16, 42, 0.78, 0.77, 0.33),
        _run_with_strict("dora", "cs170k", 16, 1, 0.78, 0.80, 0.79),
        _run_with_strict("dora", "cs170k", 16, 2, 0.63, 0.78, 0.77),
    ]
    out = summarize.render(summarize.aggregate(runs), zero_shot=None)
    assert "genmatch (strict)" in out
    assert "genmatch_strict" in out  # the per-regime gap table heading


def test_render_omits_strict_when_absent():
    runs = [_run("lora", "boolq", 8, 42, 0.88, 0.90)]
    out = summarize.render(summarize.aggregate(runs), zero_shot=None)
    assert "genmatch (strict)" not in out
    assert "genmatch_strict" not in out
