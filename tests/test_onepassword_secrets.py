"""Unique carried headless-op regression; common tests live in tests/agent/."""
from unittest import mock

from agent.secret_sources import onepassword as op


def _ok(value):
    return mock.Mock(returncode=0, stdout=value, stderr="")


def test_fetch_child_env_passes_load_desktop_app_settings(monkeypatch, tmp_path):
    """Keep the carried headless-op regression; the runtime fix is already native."""
    fake_op = tmp_path / "op"
    fake_op.write_text("")
    monkeypatch.setenv("OP_SERVICE_ACCOUNT_TOKEN", "ops_tok")
    monkeypatch.setenv("OP_LOAD_DESKTOP_APP_SETTINGS", "false")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["env"] = kwargs["env"]
        return _ok("v")

    monkeypatch.setattr(op.subprocess, "run", fake_run)
    secrets, warnings = op.fetch_onepassword_secrets(
        references={"K": "op://V/I/F"}, binary=fake_op,
        home_path=tmp_path, use_cache=False,
    )
    assert secrets == {"K": "v"} and warnings == []
    assert captured["env"]["OP_LOAD_DESKTOP_APP_SETTINGS"] == "false"
