# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer

from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

class Predictor:
    def __init__(self):
        cfg = self.init_cfg()
        self.layout_predictor = DefaultPredictor(cfg)
        self.ocr = None

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

    def layout_predict(self, image):
        image = cv2.imread(filename)
        output = self.layout_predictor(image)
        return image, output

    def ocr(self, text_image):
        pass # TODO: Run OCR to extract text

if __name__ == "__main__":
    predictor = Predictor()
    register_coco_instances("pdf_layout_val", {},
        "/mnt/LSTA6/data/nishimura/DocLayNet/COCO/val.json",
        "/mnt/LSTA6/data/nishimura/DocLayNet/PNG")
    
    dataset_dicts = DatasetCatalog.get("pdf_layout_val")
    val_layout_metadata = MetadataCatalog.get("pdf_layout_val")

    for i, d in enumerate(dataset_dicts[:5]):
        filename = d["file_name"]
        image, output = predictor.predict(filename)
