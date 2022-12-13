from layout_predictor import LayoutPredictor
from pdf_analyzer import PDFAnalyzer
from coordinate_helper import (
    convert_bbox_mediabox, 
    convert_textlines_to_raw_img_size,                        
    convert_layouts_to_raw_img_size,
    compute_IoU
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
        self.color_dicts = {cat:color for cat, color in zip(self.categories, self.colors)}
        self.pred_image_size = (1025, 1025) # paper definition

    def integrate_layout_and_textlines(self, layout, textlines, image_size):
        layout, textlines = self.scale_raw_img_size(layout, textlines, image_size)
        pred_boxes = layout['instances'].pred_boxes
        pred_categories = layout['instances'].pred_classes

        # convert layout / text to output form
        layout_json = self.jsonify_layout(pred_boxes, pred_categories)
        text_json = self.jsonify_textlines(textlines)

        # compute inclusion relationship
        layout_json, text_json = self.compute_inclusion_relation(layout_json, text_json)
        return layout_json, text_json
    
    def compute_inclusion_relation(self, layout_json, text_json):
        # layout -> texts
        for l_name, layout in layout_json.items():
            l_coordinate, category = layout["coordinate"], layout["category"]
            layout["inclusion_texts"] = {}
            for t_name, text in text_json.items():
                text.setdefault("inclusion_layouts", {})
                t_coordinate = text["coordinate"]
                iou = compute_IoU(l_coordinate, t_coordinate)
                if iou > 0:
                    layout["inclusion_texts"][t_name] = iou
                    text["inclusion_layouts"][l_name] = iou
        return layout_json, text_json

    def scale_raw_img_size(self, layout, textlines, image_size):
        layout = convert_layouts_to_raw_img_size(layout, self.pred_image_size, image_size)
        textlines = convert_textlines_to_raw_img_size(textlines, image_size)
        return layout, textlines

    def jsonify_textlines(self, textlines):
        output = {}
        for i, text in enumerate(textlines):
            output["text_" + str(i)] = {
                'coordinate': text['bbox'],
                'text': text['text']
            }
        return output

    def jsonify_layout(self, pred_boxes, pred_categories):
        output = {}
        for i, (pred_box, pred_category) in enumerate(zip(pred_boxes, pred_categories)):
            category_name = self.categories[pred_category.item()]
            pred_box = pred_box.tolist()
            output[category_name + "_" + str(i)] = {
                'coordinate': pred_box,
                'category': category_name,
            }
        return output

    def analyze(self, filename):
        pdf_images = convert_from_path(filename)

        # layout prediction / extract text-line bboxes for PDF
        output_json = {}
        for i, pdf_image in enumerate(pdf_images):
            image_size = pdf_image.size
            layout = self.layout_predictor.predict(pdf_image)
            textlines = self.pdf_analyzer.extract_textlines(filename, page_num=i)
            
            # TODO: integrate results
            layout_json, text_json = self.integrate_layout_and_textlines(layout, textlines, image_size)
            output_json["page_" + str(i)] = {
                "texts" : text_json,
                "layout" : layout_json
            }
            visualize(text_json, layout_json, pdf_image, self.color_dicts)
        return output_json

if __name__ == "__main__":
    bacon = bacon()
    bacon.analyze("../test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")