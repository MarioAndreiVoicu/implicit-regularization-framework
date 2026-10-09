import importlib
import sys
from types import SimpleNamespace


class FakeSidebar:
    def __init__(self, calls):
        self.calls = calls

    def title(self, text):
        self.calls.append(("title", text))

    def radio(self, label, options, key):
        self.calls.append(("radio", label, tuple(options), key))


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.calls = []
        self.sidebar = FakeSidebar(self.calls)


def _load_shell_with_fake_streamlit(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("framework.ui.shell", None)
    shell = importlib.import_module("framework.ui.shell")
    return shell, fake_st


def test_apply_navigation_handoff_defers_nav_page_update(monkeypatch):
    shell, fake_st = _load_shell_with_fake_streamlit(monkeypatch)

    shell.apply_navigation_handoff({"nav_page": "Compare", "free_compare_selected_runs": ["a", "b"]})

    assert fake_st.session_state["_pending_nav_page"] == "Compare"
    assert "nav_page" not in fake_st.session_state
    assert fake_st.session_state["free_compare_selected_runs"] == ["a", "b"]


def test_render_sidebar_navigation_consumes_pending_page_before_radio(monkeypatch):
    shell, fake_st = _load_shell_with_fake_streamlit(monkeypatch)
    fake_st.session_state["_pending_nav_page"] = "Compare"

    page = shell.render_sidebar_navigation()

    assert page == "Compare"
    assert fake_st.session_state["nav_page"] == "Compare"
    assert "_pending_nav_page" not in fake_st.session_state
    assert fake_st.calls[-1] == ("radio", "Go to", tuple(shell.PAGES), "nav_page")
