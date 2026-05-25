"""Tests for dora_mini.configs — pure dict-building, runs on Mac."""
import yaml

from dora_mini import configs


def test_all_configs_count():
    # 2 methods x 3 ranks x 3 seeds x 2 phases = 36
    assert len(configs.all_configs()) == 36


def test_run_id_schema_and_uniqueness():
    cfgs = configs.all_configs()
    assert "dora_mistral7b_cs170k_r8_s42" in cfgs
    assert "lora_mistral7b_boolq_r4_s1" in cfgs
    assert len(set(cfgs)) == 36


def test_alpha_is_twice_rank():
    for cfg in configs.all_configs().values():
        assert cfg["peft"]["alpha"] == 2 * cfg["peft"]["r"]


def test_phase_recipes():
    cfgs = configs.all_configs()
    boolq = cfgs["lora_mistral7b_boolq_r8_s42"]
    cs = cfgs["lora_mistral7b_cs170k_r8_s42"]
    assert boolq["data"]["train_dataset"] == "boolq"
    assert boolq["training"]["learning_rate"] == 5.0e-5
    assert boolq["training"]["num_steps"] == 500
    assert cs["data"]["train_dataset"] == "commonsense_170k"
    assert cs["training"]["learning_rate"] == 2.0e-4
    assert cs["data"]["train_size"] is None


def test_run_group_routes_by_regime():
    assert configs.run_group("lora_mistral7b_boolq_r8_s42") == "tier2-boolq"
    assert configs.run_group("dora_mistral7b_cs170k_r4_s1") == "tier2-cs170k"
    assert configs.run_group("dora_mistral7b_boolq_r8") == "tier1"


def test_yaml_roundtrip_keeps_lr_a_float(tmp_path):
    # YAML 1.1 parses bare 5e-5 as a string; the generator must avoid that.
    configs.write_configs(tmp_path)
    files = sorted(tmp_path.rglob("*.yaml"))
    assert len(files) == 36
    assert {p.parent.name for p in files} == {"tier2-boolq", "tier2-cs170k"}
    loaded = yaml.safe_load(files[0].read_text())
    assert isinstance(loaded["training"]["learning_rate"], float)
    assert loaded["output"]["run_id"] == files[0].stem


# ---------------------------------------------------------------------------
# Tier-3 sweep tests — verify the new recipe + routing, and that the Tier-2
# regex update did not regress existing Tier-2 / Tier-1 routing.
# ---------------------------------------------------------------------------


def test_run_group_routes_tier3_runs():
    # Routing depends only on the `_t3` tail, not seed value — but use real
    # sweep seeds in the fixtures to keep the test self-documenting.
    assert configs.run_group("lora_mistral7b_cs170k_r4_s114_t3") == "tier3-cs170k"
    assert configs.run_group("dora_mistral7b_cs170k_r16_s1919_t3") == "tier3-cs170k"


def test_run_group_tier2_unchanged_after_t3_regex():
    # The regex now has an optional (_t3)? group at the tail. Confirm Tier-2 and
    # Tier-1 IDs without the suffix still route exactly as before.
    assert configs.run_group("lora_mistral7b_boolq_r8_s42") == "tier2-boolq"
    assert configs.run_group("dora_mistral7b_cs170k_r4_s1") == "tier2-cs170k"
    assert configs.run_group("dora_mistral7b_boolq_r8") == "tier1"


def test_all_tier3_configs_count():
    # 2 methods x 3 ranks x 4 seeds x 1 phase (cs170k) = 24
    assert len(configs.all_tier3_configs()) == 24


def test_tier3_run_ids_all_carry_t3_suffix():
    for run_id in configs.all_tier3_configs():
        assert run_id.endswith("_t3"), run_id


def test_tier3_recipe_values():
    cfgs = configs.all_tier3_configs()
    cs = cfgs["lora_mistral7b_cs170k_r8_s114_t3"]
    assert cs["training"]["num_steps"] == 10000
    assert cs["training"]["warmup_steps"] == 400
    assert cs["training"]["eval_every"] == 400
    assert cs["training"]["learning_rate"] == 2.0e-4
    assert cs["data"]["train_dataset"] == "commonsense_170k"
    assert cs["data"]["eval_size"] == 3270
    assert cs["data"]["train_size"] is None


def test_tier3_seed_set_matches_recipe():
    seeds = {cfg["training"]["seed"] for cfg in configs.all_tier3_configs().values()}
    assert seeds == {114, 514, 1919, 810}


def test_tier3_alpha_is_twice_rank():
    for cfg in configs.all_tier3_configs().values():
        assert cfg["peft"]["alpha"] == 2 * cfg["peft"]["r"]


def test_tier3_does_not_pollute_all_configs():
    # all_configs() must remain the 36-run Tier-2 grid; Tier 3 lives in a
    # parallel function so existing call sites are untouched.
    assert len(configs.all_configs()) == 36
    for run_id in configs.all_configs():
        assert not run_id.endswith("_t3"), run_id


def test_write_tier3_configs_yaml(tmp_path):
    configs.write_tier3_configs(tmp_path)
    files = sorted(tmp_path.rglob("*.yaml"))
    assert len(files) == 24
    assert {p.parent.name for p in files} == {"tier3-cs170k"}
    loaded = yaml.safe_load(files[0].read_text())
    assert isinstance(loaded["training"]["learning_rate"], float)
    assert loaded["output"]["run_id"] == files[0].stem
    assert loaded["output"]["run_id"].endswith("_t3")
