from layout_predictor import LayoutPredictor
from pdf_analyzer import PDFAnalyzer
from coordinate_helper import (
    convert_bbox_mediabox, 
    convert_textlines_to_raw_img_size,                        
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

    def integrate_chars_with_layouts(self, layout, textlines, image_size):
        layout, textlines = self.scale_raw_img_size(layout, textlines, image_size)

        pred_boxes = layout['instances'].pred_boxes
        pred_categories = layout['instances'].pred_classes

        return layout, textlines
        """
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
        """

    def scale_raw_img_size(self, layout, textlines, image_size):
        layout = convert_layouts_to_raw_img_size(layout, self.pred_image_size, image_size)
        textlines = convert_textlines_to_raw_img_size(textlines, image_size)
        return layout, textlines

    def analyze(self, filename):
        pdf_images = convert_from_path(filename)

        # layout prediction / extract text-line bboxes for PDF
        for i, pdf_image in enumerate(pdf_images):
            image_size = pdf_image.size
            layout = self.layout_predictor.predict(pdf_image)
            textlines = self.pdf_analyzer.extract_textlines(filename, page_num=i)
            
            # TODO: integrate results
            layout, textlines = self.integrate_chars_with_layouts(layout, textlines, image_size)
            visualize(textlines, layout, pdf_image, self.categories, self.colors)

if __name__ == "__main__":
    bacon = bacon()
    bacon.analyze("../test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")