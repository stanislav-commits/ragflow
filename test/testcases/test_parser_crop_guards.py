import re

from PIL import Image

from deepdoc.parser.pdf_parser import RAGFlowPdfParser
from rag.nlp import _normalize_pdf_crop_positions, tokenize_chunks


def test_normalize_pdf_crop_positions_flattens_and_swaps_inverted_bounds():
    positions = _normalize_pdf_crop_positions([([0, 1], 18, 4, 42, 12)])

    assert positions == [
        (0, 4.0, 18.0, 12.0, 42.0),
        (1, 4.0, 18.0, 12.0, 42.0),
    ]


def test_tokenize_chunks_falls_back_to_text_only_when_crop_fails():
    class ExplodingParser:
        def crop(self, *_args, **_kwargs):
            raise ValueError("Coordinate 'lower' is less than 'upper'")

        def extract_positions(self, _text):
            return [([0], 10, 2, 50, 20)]

        def remove_tag(self, text):
            return re.sub(r"@@[0-9-]+\t[0-9.\t]+##", "", text).strip()

    chunks = ["Recovered chunk@@1\t10\t2\t50\t20##"]
    doc = {"docnm_kwd": "demo.pdf"}

    res = tokenize_chunks(chunks, doc, eng=True, pdf_parser=ExplodingParser())

    assert len(res) == 1
    assert res[0]["content_with_weight"] == "Recovered chunk"
    assert res[0]["page_num_int"] == [1]
    assert res[0]["position_int"] == [(1, 2, 10, 20, 50)]


def test_safe_crop_box_normalizes_inverted_coordinates():
    parser = RAGFlowPdfParser.__new__(RAGFlowPdfParser)
    image = Image.new("RGB", (100, 80), "white")

    crop_box = parser._safe_crop_box(image, 40, 70, 10, 20, context="unit-test")

    assert crop_box == (10, 20, 40, 70)
