import xml.etree.ElementTree as ET
from pathlib import Path

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    masks = []
    for image in root.findall('image'):
        image_id = image.get('id')
        image_name = image.get('name')
        for mask in image.findall('mask'):
            label = mask.get('label')
            rle = mask.get('rle')
            masks.append({'label': label, 'rle': rle, 'image_id': image_id, 'image_name': image_name})
    return masks

def adjust_rle(rle, tile_x, tile_y, tile_width, tile_height):
    # 示例逻辑，需要根据实际RLE格式进行调整
    adjusted_rle = rle  # 实际逻辑可能会更复杂
    return adjusted_rle

def create_annotation_xml(masks, output_path):
    annotation = ET.Element('annotation')
    for mask in masks:
        mask_element = ET.SubElement(annotation, 'mask')
        label = ET.SubElement(mask_element, 'label')
        label.text = mask['label']
        # 添加更多的细节，如调整后的RLE
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def split_annotations(annotation_path, output_folder, tile_size_x, tile_size_y):
    original_masks = parse_annotation(annotation_path)
    for tile_path in Path(output_folder).glob('tile_*.tif'):
        _, col, row = tile_path.stem.split('_')
        tile_x, tile_y = int(col) * tile_size_x, int(row) * tile_size_y
        adjusted_masks = []
        for mask in original_masks:
            adjusted_rle = adjust_rle(mask['rle'], tile_x, tile_y, tile_size_x, tile_size_y)
            adjusted_masks.append({'label': mask['label'], 'rle': adjusted_rle})
        if adjusted_masks:
            output_xml_path = tile_path.with_suffix('.xml')
            create_annotation_xml(adjusted_masks, output_xml_path)
