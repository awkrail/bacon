def convert_textlines_to_raw_img_size(textlines, image_size):
    return [{
        "mediabox": textline["mediabox"],
        "bbox": convert_bbox_mediabox(textline["bbox"], textline["mediabox"], image_size),
        "text": textline["text"]
    } for textline in textlines]

def convert_layouts_to_raw_img_size(layout, pred_image_size, image_size):
    layout['instances'].pred_boxes.tensor[:, 0] *= (image_size[0]/pred_image_size[0])
    layout['instances'].pred_boxes.tensor[:, 1] *= (image_size[1]/pred_image_size[1])
    layout['instances'].pred_boxes.tensor[:, 2] *= (image_size[0]/pred_image_size[0])
    layout['instances'].pred_boxes.tensor[:, 3] *= (image_size[1]/pred_image_size[1])
    return layout

def convert_bbox_mediabox(bbox, mediabox, image_size):
    """
    Because the PDF's origin coordinates are bottom left, I have to convert it to upper left for image processing.
    In addition, scaling the coordinates to images is necessary.
    """
    x1, y1, x2, y2 = bbox
    x1, x2 = scale([x1, x2], (image_size[0]/mediabox[0]))
    y1, y2 = scale([y1, y2], (image_size[1]/mediabox[1]))
    return [x1, image_size[1]-y2, x2, image_size[1]-y1]
    
def scale(xy, ratio_xy):
    return [xy[0]*ratio_xy, xy[1]*ratio_xy]

def compute_overlap(pred_box, char_box):
    x1, y1, x2, y2 = pred_box.tolist()
    hat_x1, hat_y1, hat_x2, hat_y2 = char_box
    w = min(hat_x2, x2) - max(hat_x1, x1)
    h = min(hat_y2, y2) - max(hat_y1, y1)
    return max(w*h, 0)