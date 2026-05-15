"""Tests for dora_mini.paths.

These run on Mac without needing the LLM stack — they only exercise env-var
resolution and Path construction.
"""
import pytest

from dora_mini import paths


def test_project_dir_requires_env(monkeypatch):
    monkeypatch.delenv("PROJECT_DIR", raising=False)
    with pytest.raises(RuntimeError, match="PROJECT_DIR is not set"):
        paths.project_dir()


def test_project_dir_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    assert paths.project_dir() == tmp_path


def test_venv_dir_requires_env(monkeypatch):
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    with pytest.raises(RuntimeError, match="UV_PROJECT_ENVIRONMENT is not set"):
        paths.venv_dir()


def test_venv_dir_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path))
    assert paths.venv_dir() == tmp_path


def test_hf_home_requires_env(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    with pytest.raises(RuntimeError, match="HF_HOME is not set"):
        paths.hf_home()


def test_hf_home_returns_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert paths.hf_home() == tmp_path


def test_results_dir_creates_run_subdir(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    rd = paths.results_dir("foo_run")
    assert rd == tmp_path / "results" / "foo_run"
    assert rd.is_dir()


def test_results_dir_idempotent_preserves_contents(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    rd1 = paths.results_dir("bar_run")
    (rd1 / "sentinel.txt").write_text("keep")
    rd2 = paths.results_dir("bar_run")
    assert rd1 == rd2
    assert rd1.is_dir()
    assert (rd2 / "sentinel.txt").read_text() == "keep"


def test_results_dir_rejects_empty_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="run_id must not be empty"):
        paths.results_dir("")
    with pytest.raises(ValueError, match="run_id must not be empty"):
        paths.results_dir("   ")
