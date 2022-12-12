import os, cv2
import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Other library
from PIL import Image

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
    
    def predict(self, pdf_image):
        pdf_image = pdf_image.convert("RGB").resize(self.input_img_size)
        bgr_image = LayoutPredictor.convert_rgb_to_bgr(pdf_image)
        layout = self.layout_predictor(bgr_image)
        return layout

    @staticmethod
    def convert_rgb_to_bgr(image):
        return np.array(image)[:, :, ::-1]