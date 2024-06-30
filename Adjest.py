import xml.etree.ElementTree as ET
from pathlib import Path

def parse_annotation(xml_file):
    print(f"Trying to parse XML file: {xml_file}")
    tree = ET.parse(xml_file)
    root = tree.getroot()
    masks = []
    for image in root.findall('image'):
        image_id = image.get('id')
        image_name = image.get('name')
        for mask in image.findall('mask'):
            label = mask.get('label')
            rle = mask.get('rle')
            # 这里我们只收集掩码的基本信息，实际的RLE处理将在后续步骤中进行
            masks.append({'label': label, 'rle': rle, 'image_id': image_id, 'image_name': image_name})
    print(f"Parsed masks: {masks}")
    return masks

def adjust_mask(mask, tile_x, tile_y, tile_width, tile_height):
    # 这里需要实现根据分割后的图像调整RLE掩码的逻辑
    # 这可能包括RLE掩码的解码、坐标调整、重新编码等步骤
    adjusted_mask = mask  # 示例代码，实际需要替换为具体逻辑
    print(f"Adjusted mask for tile position ({tile_x}, {tile_y})")
    return adjusted_mask

def create_annotation_xml(masks, output_path):
    print(f"Creating XML file at: {output_path} with masks: {masks}")
    annotation = ET.Element('annotation')
    for mask in masks:
        mask_element = ET.SubElement(annotation, 'mask')
        label = ET.SubElement(mask_element, 'label')
        label.text = mask['label']
        # 此处省略了RLE数据的处理，需要根据实际逻辑添加
    tree = ET.ElementTree(annotation)
    tree.write(output_path)

def split_annotations(annotation_path, output_folder, tile_size_x, tile_size_y):
    original_masks = parse_annotation(annotation_path)
    for tile_path in Path(output_folder).glob('tile_*.tif'):
        # Corrected to match the expected tile filename pattern
        _, col, row = tile_path.stem.split('_')
        tile_x, tile_y = int(col), int(row)
        adjusted_masks = []
        for mask in original_masks:
            # Placeholder for actual logic to adjust the mask based on tile position
            adjusted_mask = adjust_mask(mask, tile_x, tile_y, tile_size_x, tile_size_y)
            if adjusted_mask:
                adjusted_masks.append(adjusted_mask)
        if adjusted_masks:
            output_xml_path = tile_path.with_suffix('.xml')
            create_annotation_xml(adjusted_masks, output_xml_path)



# Adjust these paths as necessary
annotation_path = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\Text2\annotations.xml'
output_folder = r'C:\Simon\Master of Professional Engineering\Software Design Project\Image Dataset\Text2\segment'
tile_size_x = 256  # Tile width in pixels
tile_size_y = 256  # Tile height in pixels

split_annotations(annotation_path, output_folder, tile_size_x, tile_size_y)
