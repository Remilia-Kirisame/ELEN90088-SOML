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
