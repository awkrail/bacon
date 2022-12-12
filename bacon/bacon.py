from layout_predictor import LayoutPredictor
from pdf_analyzer import PDFAnalyzer
from coordinate_helper import (
    convert_bbox_mediabox, 
    convert_char_bboxes_to_raw_img_size,                        
    convert_layouts_to_raw_img_size,
    compute_overlap
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
        self.overlap_threshold = 0.8

    def is_text_category(self, category_id):
        return self.categories[category_id.item()] in ['Caption', 'Footnote', 'List-item', 'Page-footer',
                                                       'Page-header', 'Section-header', 'Table', 'Text', 'Title']

    def is_overlapped_threshold(self, pred_box, char_box):
        char_area = (char_box[2] - char_box[0]) * (char_box[3] - char_box[1])
        overlapped_area = compute_overlap(pred_box, char_box)
        return overlapped_area/char_area < self.overlap_threshold

    def group_up_with_box(self, pred_box, char_boxes):
        outputs = []

        # filter overlapped char boxes to pred_box
        no_overlapped_chars = [(char, char_box) for char, char_box in char_boxes if self.is_overlapped_threshold(pred_box, char_box)]
        return no_overlapped_chars

    def integrate_chars_with_layouts(self, bbox_dict_list, layouts, image_sizes):
        char_boxes_list, layouts = self.scale_raw_img_size(bbox_dict_list, layouts, image_sizes)
        outputs = []
        for char_boxes, layout in zip(char_boxes_list, layouts):
            pred_boxes = layout['instances'].pred_boxes
            pred_categories = layout['instances'].pred_classes

            for pred_box, pred_category in zip(pred_boxes, pred_categories):
                if self.is_text_category(pred_category):
                    box_output = self.group_up_with_box(pred_box, char_boxes)
                    return box_output
                    #outputs.append(box_output)
                else:
                    box_output = {
                        'coordinate': pred_box.tolist(),
                        'category': self.categories[pred_category.item()],
                        'text': None
                    }
                    outputs.append(box_output)
        return outputs

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
        #char_bboxes, layouts = self.integrate_chars_with_layouts(bbox_dict_list, layouts, image_sizes)
        no_overlapped_chars = self.integrate_chars_with_layouts(bbox_dict_list, layouts, image_sizes)
        
        visualize(no_overlapped_chars, layouts[0], pdf_images[0], self.categories, self.colors)
        #output = self.jsonify(layouts_with_texts)
        #return output

if __name__ == "__main__":
    bacon = bacon()
    bacon.analyze("../test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")