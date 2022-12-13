from PIL import Image, ImageDraw

def compute_center(coordinate):
    x1, y1, x2, y2 = coordinate
    center_x = (x2 + x1) / 2
    center_y = (y2 + y1) / 2
    return center_x, center_y

def visualize(text_json, layout_json, pdf_image, color_dict):
    """
    visualize single PDF file
    """
    draw = ImageDraw.Draw(pdf_image)
    # add layout
    for l_name, layout in layout_json.items():
        color = color_dict[l_name.split("_")[0]]
        center_x, center_y = compute_center(layout["coordinate"])
        draw.rectangle(layout["coordinate"], outline=color, width=5)
        draw.text((center_x, center_y), l_name, 'red')

    for t_name, text in text_json.items():
        draw.rectangle(text["coordinate"], outline=(0,0,0), width=3)
    pdf_image.save("./test.png")
