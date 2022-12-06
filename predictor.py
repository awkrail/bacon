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
from detectron2.data.detection_utils import read_image, convert_image_to_rgb
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

from PIL import Image

class Predictor:
    def __init__(self):
        cfg = self.init_cfg()
        self.layout_predictor = DefaultPredictor(cfg)

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
        pass

    def layout_predict(self, image):
        image = read_image(filename, format="BGR")
        layout = self.layout_predictor(image)
        return image, layout

    def ocr(self, text_image):
        characters = self.ocr_reader.readtext(text_image)
        return characters

def is_target_class(category):
    return category in ['Caption', 'Footnote', 'Page-footer',
                        'Page-header', 'Section-header', 'Text', 'Title']


if __name__ == "__main__":
    predictor = Predictor()
    register_coco_instances("pdf_layout_val", {},
        "/mnt/LSTA6/data/nishimura/DocLayNet/COCO/val.json",
        "/mnt/LSTA6/data/nishimura/DocLayNet/PNG")
    
    dataset_dicts = DatasetCatalog.get("pdf_layout_val")
    val_layout_metadata = MetadataCatalog.get("pdf_layout_val")

    classes = val_layout_metadata.thing_classes

    for i, d in enumerate(dataset_dicts[:5]):
        filename = d["file_name"]
        image, layout = predictor.layout_predict(filename)
        
        pred_boxes = layout['instances'].pred_boxes
        categories = layout['instances'].pred_classes

        output_json = {}

        for j, (pred_box, category) in enumerate(zip(pred_boxes, categories)):
            x1, y1, x2, y2 = [int(i) for i in pred_box.tolist()]
            if is_target_class(classes[category.item()]):
                cropped_img = image[y1:y2, x1:x2]
                import ipdb; ipdb.set_trace()
                image = convert_image_to_rgb(image)
                Image.fromarray(image).save("./test.png")
                #cv2.imwrite("images/image_{}_{}.png".format(i, j), cropped_img)
                #text_filename = "images/image_{}_{}.png".format(i, j)
                #text = predictor.ocr(text_filename)
                #output_json['']

