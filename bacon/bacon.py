from layout_predictor import LayoutPredictor
from pdf_analyzer import PDFAnalyzer

# visualize
from PIL import ImageDraw, Image

class bacon:
    def __init__(self):
        self.layout_predictor = LayoutPredictor()
        self.pdf_analyzer = PDFAnalyzer()
        self.categories = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
                           "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]
        self.colors = [(255,0,0), (0,0,255), (0,255,0), (255,255,0), (0,255,255),
                       (255,0,255), (128,128,0), (0,128,128), (128,0,128), (128,0,0),
                       (0,0,128)]

    def analyze(self, file_name):
        img_layout_pairs = self.layout_predictor.predict(file_name)
        _, _, raw_image_size = img_layout_pairs[0] # adhoc
        char_bbox_list = self.pdf_analyzer.analyze(file_name, raw_image_size)
        return img_layout_pairs, char_bbox_list

    def visualize(self, img_layout_pairs, char_bbox_list):
        for image_layout in img_layout_pairs:
            image, layout, raw_image_size = image_layout
            draw = ImageDraw.Draw(image)

            # first, layout prediction
            pred_bboxes = layout['instances'].pred_boxes
            pred_categories = layout['instances'].pred_classes
            
            for bbox, category in zip(pred_bboxes, pred_categories):
                draw.rectangle(bbox.tolist(), outline=self.colors[category.item()])    
            raw_sized_image = image.resize(raw_image_size)
            raw_sized_image.save("./raw_sized_image.png")

        raw_sized_image = Image.open("./raw_sized_image.png")
        draw = ImageDraw.Draw(raw_sized_image)
        for char, bbox in char_bbox_list:
            draw.rectangle(bbox, outline=(0,0,0))
        raw_sized_image.save("./raw_sized_with_chars.png")

if __name__ == "__main__":
    bacon = bacon()
    img_layout_pairs, char_bbox_list = bacon.analyze("../test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")
    bacon.visualize(img_layout_pairs, char_bbox_list)