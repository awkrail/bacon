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
    
    def analyze(self, filename, image_size):
        bbox_list = []
        with open(filename, "rb") as fb:
            for page in PDFPage.get_pages(fb):
                self.interpreter.process_page(page)
                for char in self.device.get_characters():
                    text, bbox = char
                    bbox = PDFAnalyzer.convert_coordinates(bbox, page.mediabox[2:], image_size) # page.mediabox = [0, 0, w, h]
                    bbox_list.append([text, bbox])
        return bbox_list

    @staticmethod
    def convert_coordinates(bbox, mediabox, image_size):
        """
        Because the PDF's origin coordinates are bottom left, I have to convert it to upper left for image processing.
        In addition, scaling the coordinates to images is necessary.
        """
        x1, y1, x2, y2 = bbox
        x1, x2 = PDFAnalyzer.scale([x1, x2], (image_size[0]/mediabox[0]))
        y1, y2 = PDFAnalyzer.scale([y1, y2], (image_size[1]/mediabox[1]))
        return [x1, image_size[1]-y1, x2, image_size[1]-y2]
    
    @staticmethod
    def scale(axis, ratio):
        return [axis[0]*ratio, axis[1]*ratio]