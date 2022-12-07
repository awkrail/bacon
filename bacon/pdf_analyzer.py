from collections.abc import Iterable

from typing import Iterable, Any

from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.layout import LTChar, LTFigure, LTImage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.high_level import extract_pages

from pdf2image import convert_from_path
from PIL import ImageDraw

class PageAnalyzer(PDFLayoutAnalyzer):
    def __init__(self, rsrcmgr):
        super().__init__(rsrcmgr)
        self._characters = []

    def get_characters(self):
        return self._characters

    def receive_layout(self, ltpage):
        stack = [ltpage]
        while len(stack) > 0:
            item = stack.pop()

            if isinstance(item, LTChar):
                self._characters.append([item.get_text(), item.bbox])

            if isinstance(item, Iterable):
                stack.extend(list(iter(item)))


class PDFAnalyzer:
    def __init__(self):
        self.rsrcmgr = PDFResourceManager()
        self.device = PageAnalyzer(self.rsrcmgr)
        self.interpreter = PDFPageInterpreter(self.rsrcmgr, self.device)

    def extract_char_bbox(self, filename):
        bbox_dict_list = []
        with open(filename, "rb") as fb:
            for page in PDFPage.get_pages(fb):
                bbox_dict = { "mediabox" : page.mediabox[2:], "bboxes": [] }
                self.interpreter.process_page(page)
                for char in self.device.get_characters():
                    text, bbox = char
                    bbox_dict["bboxes"].append([text, bbox])
                bbox_dict_list.append(bbox_dict)
        return bbox_dict_list