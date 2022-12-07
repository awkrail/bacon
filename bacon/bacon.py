from layout_predictor import LayoutPredictor
from pdf_analyzer import PDFAnalyzer
from coordinate_helper import (
    convert_bbox_mediabox, 
    convert_char_bboxes_to_raw_img_size,                        
    convert_layouts_to_raw_img_size
)
from visualizer import visualize

# visualize
from PIL import ImageDraw, Image
from pdf2image import convert_from_path

class bacon:
    def __init__(self):
        self.layout_predictor = LayoutPredictor()
        self.pdf_analyzer = PDFAnalyzer()
        self.categories = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
                           "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
        self.colors = [(255,0,0), (0,0,255), (0,255,0), (255,255,0), (0,255,255),
                       (255,0,255), (128,128,0), (0,128,128), (128,0,128), (128,0,0),
                       (0,0,128)]
        self.pred_image_size = (1025, 1025) # paper definition

    def integrate_chars_with_layouts(self, bbox_dict_list, layouts, image_sizes):
        char_boxes_list, layouts = self.scale_raw_img_size(bbox_dict_list, layouts, image_sizes)
        return char_boxes_list, layouts

    def scale_raw_img_size(self, bbox_dict_list, layouts, image_sizes):
        scaled_char_boxes_list, scaled_layouts = [], []
        for bbox_dict, layout, image_size in zip(bbox_dict_list, layouts, image_sizes):
            char_bboxes = convert_char_bboxes_to_raw_img_size(bbox_dict, image_size)
            layout = convert_layouts_to_raw_img_size(layout, self.pred_image_size, image_size)
            scaled_char_boxes_list.append(char_bboxes)
            scaled_layouts.append(layout)
        return scaled_char_boxes_list, scaled_layouts

    def analyze(self, filename):
        pdf_images = convert_from_path(filename)
        image_sizes = [image.size for image in pdf_images]
        
        # layout prediction / extract char-level bboxes from PDF
        layouts = self.layout_predictor.predict(pdf_images)
        bbox_dict_list = self.pdf_analyzer.extract_char_bbox(filename)

        # integrate the results
        char_bboxes, layouts = self.integrate_chars_with_layouts(bbox_dict_list, layouts, image_sizes)
        visualize(char_bboxes[0], layouts[0], pdf_images[0], self.categories, self.colors)
        #output = self.jsonify(layouts_with_texts)
        #return output

if __name__ == "__main__":
    bacon = bacon()
    bacon.analyze("../test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")