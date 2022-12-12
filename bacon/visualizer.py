from PIL import Image, ImageDraw

def visualize(textlines, layout, pdf_image, categories, colors):
    """
    visualize single PDF file
    """
    draw = ImageDraw.Draw(pdf_image)
    # add layout
    pred_boxes = layout['instances'].pred_boxes
    pred_categories = layout['instances'].pred_classes
    for pred_category, pred_box in zip(pred_categories, pred_boxes):
        cat_name = categories[pred_category.item()]
        draw.rectangle(pred_box.tolist(), outline=colors[pred_category.item()], width=5)

    for textline in textlines:
        bbox = textline["bbox"]
        draw.rectangle(bbox, outline=(0,0,0), width=3)
    pdf_image.save("./test.png")
