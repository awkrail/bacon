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

if __name__ == "__main__":
    register_coco_instances("pdf_layout_train", {}, "/mnt/LSTA6/data/nishimura/DocLayNet/COCO/train.json", "/mnt/LSTA6/data/nishimura/DocLayNet/PNG")
    register_coco_instances("pdf_layout_val", {}, "/mnt/LSTA6/data/nishimura/DocLayNet/COCO/val.json", "/mnt/LSTA6/data/nishimura/DocLayNet/PNG")

    layout_metadata = MetadataCatalog.get("pdf_layout_train")
    dataset_dicts = DatasetCatalog.get("pdf_layout_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pdf_layout_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    cfg.SOLVER.MAX_ITER = 100000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11
    cfg.OUTPUT_DIR = "/mnt/LSTA6/data/nishimura/DocLayNet/Models"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)

    val_layout_metadata = MetadataCatalog.get("pdf_layout_val")
    dataset_dicts = DatasetCatalog.get("pdf_layout_val")

    for d in random.sample(dataset_dicts, 1):
        image = cv2.imread(d["file_name"])
        outputs = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata=val_layout_metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        cv2.imwrite("predict.jpg", out.get_image()[:, :, ::-1])

    evaluator = COCOEvaluator("pdf_layout_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "pdf_layout_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))