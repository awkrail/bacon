import os, cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Other library
from PIL import Image
from pdf2image import convert_from_path

# Check the results (tmp)
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances

class Predictor:
    def __init__(self):
        cfg = self.init_cfg()
        self.layout_predictor = DefaultPredictor(cfg)
        # They are defined in the original paper
        # See: https://arxiv.org/abs/2206.01062
        self.input_img_size = ((1025, 1025))
        self.categories = ["Caption", "Footnote", "Formula", "List-item", "Page-footer",
                           "Page-header", "Picture", "Section-header", "Table", "Text", "Title"]

    def init_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.OUTPUT_DIR = "/mnt/LSTA6/data/nishimura/DocLayNet/Models"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0054599.pth")
        return cfg
    
    def predict(self, pdf_filename):
        # TODO: first, adopt the method to 1-page PDF
        pdf_images = convert_from_path(pdf_filename)
        for pdf_image in pdf_images:
            raw_image_size = pdf_image.size
            tgt_image, layout = self.layout_predict(pdf_image)
            return tgt_image, layout
            

    def layout_predict(self, image):
        image = image.convert("RGB").resize(self.input_img_size)
        image = self.convert_rgb_to_bgr(image)
        layout = self.layout_predictor(image)
        return image, layout

    @staticmethod
    def convert_rgb_to_bgr(image):
        return np.array(image)[:, :, ::-1]

def is_target_class(category):
    return category in ['Caption', 'Footnote', 'Page-footer',
                        'Page-header', 'Section-header', 'Text', 'Title']

if __name__ == "__main__":
    predictor = Predictor()
    pdf_image, layout = predictor.predict("test_pdf_files/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.pdf")

    # check the results
    register_coco_instances("pdf_layout_val", {}, "/mnt/LSTA6/data/nishimura/DocLayNet/COCO/val.json", "/mnt/LSTA6/data/nishimura/DocLayNet/PNG")
    val_layout_metadata = MetadataCatalog.get("pdf_layout_val")
    dataset_dicts = DatasetCatalog.get("pdf_layout_val")
    image = cv2.imread("/mnt/LSTA6/data/nishimura/DocLayNet/PNG/c0dc81a3477ac31579cc4ecc7e2086d487996e344a7cd0c474528871aa5ac28b.png")
    v = Visualizer(image[:, :, ::-1], metadata=val_layout_metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(layout['instances'])
    cv2.imwrite("./test_layout.png", out.get_image()[:, :, ::-1])