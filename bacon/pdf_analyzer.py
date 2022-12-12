from collections.abc import Iterable
from pdfminer.layout import LTTextLineHorizontal, LTTextContainer
from pdfminer.high_level import extract_pages


class PDFAnalyzer:
    def __init__(self):
        pass

    def extract_textlines(self, filename, page_num):
        text_lines = []
        for page in extract_pages(filename, page_numbers=page_num):
            for element in page:
                if isinstance(element, LTTextContainer):
                    for text_line in element:
                        if isinstance(text_line, LTTextLineHorizontal):
                            text = text_line.get_text()
                            bbox = text_line.bbox
                            text_lines.append({
                                "mediabox": page.bbox[2:], # (w, h)
                                "text": text,
                                "bbox": bbox
                            })
        return text_lines