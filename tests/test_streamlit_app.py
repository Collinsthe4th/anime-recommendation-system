"""Smoke test: the Streamlit app must run without raising."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_app_runs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value.startswith("🎌 AnimRec")
    at.radio[0].set_value("A description").run()
    assert not at.exception
    at.selectbox[0].set_value("Mecha").run()
    assert not at.exception
