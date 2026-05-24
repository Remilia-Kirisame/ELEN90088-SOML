"""Tests for dora_mini.summarize_tier3 — pure stdlib, runs on Mac."""
from dora_mini import summarize, summarize_tier3


def test_parse_run_id_valid_tier3():
    got = summarize_tier3.parse_run_id("dora_mistral7b_cs170k_r8_s114_t3")
    assert got == {"method": "dora", "trainset": "cs170k", "r": 8, "seed": 114}


def test_parse_run_id_rejects_tier2():
    # Tier-2 ids (no _t3 suffix) must not match the Tier-3 regex.
    assert summarize_tier3.parse_run_id("dora_mistral7b_cs170k_r8_s42") is None


def test_parse_run_id_rejects_boolq_tier3():
    # Tier-3 is cs170k-only by recipe; a hypothetical boolq+_t3 should not match.
    assert summarize_tier3.parse_run_id("lora_mistral7b_boolq_r4_s114_t3") is None


def test_parse_run_id_rejects_tier1():
    assert summarize_tier3.parse_run_id("dora_mistral7b_boolq_r8") is None
    assert summarize_tier3.parse_run_id("garbage_t3") is None


def _run_t3(method, r, seed, lik, gen=None, *, runtime=300, mem=60.0, params=1000):
    """Build a synthetic Tier-3 metrics dict. gen=None means the gen-eval pass has not run yet."""
    d = {
        "run_id": f"{method}_mistral7b_cs170k_r{r}_s{seed}_t3",
        "eval_accuracy_likelihood": lik,
        "eval_loss": 1.0,
        "train_runtime_s": runtime,
        "peak_memory_gb": mem,
        "trainable_params": params,
    }
    if gen is not None:
        d["eval_accuracy_genmatch"] = gen
    return d


def test_aggregate_means_and_std_t3():
    runs = [
        _run_t3("lora", 8, 114, 0.80, 0.78),
        _run_t3("lora", 8, 514, 0.82, 0.80),
        _run_t3("lora", 8, 1919, 0.84, 0.82),
        _run_t3("lora", 8, 810, 0.78, 0.76),
    ]
    cells = summarize_tier3.aggregate(runs)
    cell = cells[("cs170k", "lora", 8)]
    assert cell["n"] == 4
    assert abs(cell["likelihood_mean"] - 0.81) < 1e-9
    assert abs(cell["genmatch_mean"] - 0.79) < 1e-9
    assert cell["likelihood_std"] > 0


def test_aggregate_tolerates_missing_genmatch_during_phase_d():
    # Mid-Phase-D state: some runs have genmatch, others don't yet.
    runs = [
        _run_t3("dora", 16, 114, 0.62, 0.85),
        _run_t3("dora", 16, 514, 0.38, None),
        _run_t3("dora", 16, 1919, 0.39, 0.83),
        _run_t3("dora", 16, 810, 0.38, None),
    ]
    cells = summarize_tier3.aggregate(runs)
    cell = cells[("cs170k", "dora", 16)]
    assert cell["n"] == 4  # likelihood always present, so n covers all
    # genmatch_mean only reflects the runs that have it
    assert cell["genmatch_n"] == 2
    assert abs(cell["genmatch_mean"] - 0.84) < 1e-9


def test_render_works_without_genmatch():
    # build_summary-equivalent code path should produce a valid markdown doc even when
    # no gen-eval results are present yet — genmatch column renders as `-`.
    runs = [_run_t3("lora", 4, s, 0.78) for s in (114, 514, 1919, 810)]
    cells = summarize_tier3.aggregate(runs)
    md = summarize_tier3.render(cells, t2_cells=None)
    assert "Tier 3 results summary" in md
    assert "LORA" in md  # method name uppercased in the per-cell table
    assert "|" in md  # has tables
    # Genmatch column should show `-` placeholder when missing.
    assert " - |" in md or "| - |" in md


def test_render_cross_tier_comparison_includes_t2_when_provided():
    t3_runs = [_run_t3("lora", 4, s, 0.78, 0.82) for s in (114, 514, 1919, 810)]
    t3_cells = summarize_tier3.aggregate(t3_runs)

    # Hand-build a small T2 cells dict via the Tier-2 aggregator to avoid disk I/O.
    def _t2_run(method, ts, r, seed, lik, gen):
        return {
            "run_id": f"{method}_mistral7b_{ts}_r{r}_s{seed}",
            "eval_accuracy_likelihood": lik,
            "eval_accuracy_genmatch": gen,
            "eval_loss": 1.0, "train_runtime_s": 100, "peak_memory_gb": 10.0,
            "trainable_params": 1000,
        }
    t2_runs = [
        _t2_run("lora", "cs170k", 4, 42, 0.80, 0.85),
        _t2_run("lora", "cs170k", 4, 1, 0.78, 0.84),
        _t2_run("lora", "cs170k", 4, 2, 0.77, 0.86),
    ]
    t2_cells = summarize.aggregate(t2_runs)
    md = summarize_tier3.render(t3_cells, t2_cells=t2_cells)
    assert "Cross-tier comparison" in md
    assert "T3 - T2" in md or "Tier-3 vs Tier-2" in md


def test_load_runs_skips_runs_without_likelihood(tmp_path, monkeypatch):
    """load_runs uses glob over results/tier3-cs170k/*/metrics.json. Verify a partial-write run is skipped."""
    import json, os
    rdir = tmp_path / "results" / "tier3-cs170k" / "dora_mistral7b_cs170k_r4_s114_t3"
    rdir.mkdir(parents=True)
    # Half-written metrics — no eval_accuracy_likelihood yet.
    (rdir / "metrics.json").write_text(json.dumps({
        "run_id": "dora_mistral7b_cs170k_r4_s114_t3",
        "train_runtime_s": 100,
    }))
    monkeypatch.chdir(tmp_path)
    runs = summarize_tier3.load_runs()
    assert runs == []  # nothing valid yet
