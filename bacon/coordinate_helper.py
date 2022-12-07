def convert_char_bboxes_to_raw_img_size(bbox_dict, image_size):
    mediabox = bbox_dict["mediabox"]
    char_bboxes = bbox_dict["bboxes"]
    char_bboxes = [(char, convert_bbox_mediabox(bbox, mediabox, image_size))for (char, bbox) in char_bboxes]
    return char_bboxes

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
    return [x1, image_size[1]-y1, x2, image_size[1]-y2]
    
def scale(xy, ratio_xy):
    return [xy[0]*ratio_xy, xy[1]*ratio_xy]