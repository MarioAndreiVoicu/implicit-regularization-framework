from pathlib import Path

from framework.ui.page_services import DocsPageService


def test_list_and_load_sections_read_markdown_docs(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "00_intro.md").write_text("# Intro\nhello with “quotes”", encoding="utf-8")
    (docs_dir / "01_more.md").write_text("# More\nworld", encoding="utf-8")

    service = DocsPageService(docs_dir=docs_dir)
    sections = service.list_sections()

    assert [section["title"] for section in sections] == ["Intro", "More"]
    intro = service.load_section("00_intro")
    assert intro["slug"] == "00_intro"
    assert "hello" in intro["content"]


def test_navigation_source_includes_docs_and_new_page_names():
    root = Path(__file__).resolve().parents[1]
    shell_source = (root / "framework/ui/shell.py").read_text(encoding="utf-8")
    assert '"Docs / Theory"' in shell_source
    assert '"Compare"' in shell_source
    assert 'PAGES = [' in shell_source
    pages_block = shell_source.split('PAGES = [', 1)[1].split(']', 1)[0]
    assert '"Run History"' not in pages_block
