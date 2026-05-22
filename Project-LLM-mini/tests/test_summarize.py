"""Tests for dora_mini.summarize — pure stdlib, runs on Mac."""
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
