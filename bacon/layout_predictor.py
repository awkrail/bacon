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


class LayoutPredictor:
    def __init__(self):
        cfg = self.init_cfg()
        self.layout_predictor = DefaultPredictor(cfg)
        # They are defined in the original paper
        # See: https://arxiv.org/abs/2206.01062
        self.input_img_size = ((1025, 1025))

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
        img_layout_pairs = []
        for pdf_image in pdf_images:
            raw_image_size = pdf_image.size
            tgt_image, layout = self.layout_predict(pdf_image)
            img_layout_pairs.append((tgt_image, layout, raw_image_size))
        return img_layout_pairs

    def layout_predict(self, image):
        image = image.convert("RGB").resize(self.input_img_size)
        bgr_image = LayoutPredictor.convert_rgb_to_bgr(image)
        layout = self.layout_predictor(bgr_image)
        return image, layout

    @staticmethod
    def convert_rgb_to_bgr(image):
        return np.array(image)[:, :, ::-1]