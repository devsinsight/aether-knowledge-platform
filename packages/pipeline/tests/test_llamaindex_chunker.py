"""Docling keeps what a page holds before its first heading once the title is its <h1>."""

from __future__ import annotations

from io import BytesIO

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import DocumentConverter

from aether_pipeline.adapters.llamaindex_chunker import with_title

# The top of ADR-015 as the wiki serves it: a status table, then the first section.
BODY = (
    '<div class="table-wrap"><table><tbody>'
    "<tr><th><p>Status</p></th><td><p>Accepted</p></td></tr>"
    "<tr><th><p>Deciders</p></th><td><p>Daniel Okafor, Mei Tanaka</p></td></tr>"
    "</tbody></table></div><h2>Context</h2><p>Every developer must run the whole game locally.</p>"
)


def markdown(body: str) -> str:
    stream = DocumentStream(name="page.html", stream=BytesIO(body.encode("utf-8")))
    converter = DocumentConverter(allowed_formats=[InputFormat.HTML])
    return converter.convert(stream).document.export_to_markdown()


def test_without_an_h1_docling_drops_the_top_of_the_page() -> None:
    assert "Accepted" not in markdown(BODY)  # the reason for with_title


def test_the_title_as_h1_keeps_the_top_of_the_page() -> None:
    text = markdown(with_title(BODY, "ADR-015 Local Development & LocalStack"))
    assert text.startswith("# ADR-015 Local Development")
    assert "Accepted" in text and "Daniel Okafor, Mei Tanaka" in text and "Every developer" in text


def test_a_page_with_its_own_h1_is_left_as_it_is() -> None:
    body = "<h1>Ships</h1><p>The player chooses one ship.</p>"
    assert with_title(body, "Ships") == body
    assert with_title(BODY, "") == BODY
