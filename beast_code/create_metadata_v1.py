import os
import yaml
import hashlib
import cv2
import torch
import layoutparser as lp
from datetime import datetime
import logging
import json
import pytesseract
from layoutparser.ocr import TesseractAgent
from argparse import ArgumentParser

# Setup logging
logging.basicConfig(filename='process.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Initialize Tesseract agent and model globally to avoid reinitialization
def initialize_tesseract(tesseract_path, tessdata_dir):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    return lp.TesseractAgent(languages='swe', config=f'--tessdata-dir "{tessdata_dir}" --psm 6')

def initialize_model(model_config_path, model_path, label_map):
    return lp.models.Detectron2LayoutModel(
        config_path=model_config_path,
        model_path=model_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
        label_map=label_map
    )

def compute_md5_checksum(file_path):
    """Compute MD5 checksum for file integrity."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            md5.update(chunk)
    return md5.hexdigest()

def process_image(image_path, model, ocr_agent, output_folder, metadata_folder, total_pages):
    try:
        image = cv2.imread(image_path)
        image = image[..., ::-1]  # Convert BGR to RGB

        layout = model.detect(image)
        ocr_text_blocks, fig_blocks = perform_ocr(image, layout, ocr_agent)

        metadata = generate_metadata(image_path, ocr_text_blocks, fig_blocks, total_pages)
        save_output_and_metadata(image_path, layout, metadata, output_folder, metadata_folder)

    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def perform_ocr(image, layout, ocr_agent):
    """Perform OCR on text blocks and return categorized blocks."""
    text_blocks = lp.Layout([b for b in layout if b.type in {'text', 'author', 'header_or_footer', 'page_no', 'title'}])
    figure_blocks = lp.Layout([b for b in layout if b.type in {'advertisement', 'image', 'list', 'table'}])

    for block in text_blocks:
        padding = 35 if block.type == 'page_no' else 7
        segment_image = block.pad(left=padding, right=padding, top=padding, bottom=padding).crop_image(image)
        text = ocr_agent.detect(segment_image)
        block.set(text=text, inplace=True)

    return text_blocks, figure_blocks

def generate_metadata(image_path, text_blocks, figure_blocks, total_pages):
    filename = os.path.basename(image_path)
    timestamp = datetime.now().isoformat()
    file_size = os.path.getsize(image_path)
    checksum = compute_md5_checksum(image_path)

    parts = filename.split('_')
    journal_name, year, volume_info, number_index, page_index = extract_file_info(parts)

    identified_elements = categorize_layout_elements(text_blocks, figure_blocks)

    metadata = {
        'Swemper-page-descriptor': {
            'PageFilename': filename,
            'Timestamp': timestamp,
            'FileType': 'JPEG',
            'FileSize': f'{file_size / (1024 * 1024):.2f}MB',
            'CreationDate': timestamp.split("T")[0],
            'md5_checksum': checksum,
            'Source': 'jordemodern(in-house scanned)',
            'Language': 'Swedish',
            'DocumentTopic': 'jordemodern-journal',
            'IdentifiedLayoutElements': identified_elements,
            'JournalName': journal_name,
            'Year': year,
            'VolumeIndex': volume_info,
            'NumberIndex': number_index,
            'PageIndex': page_index,
            'TotalPages': total_pages,
            'ContainerVolumeFile': filename.replace(".jpg", ".yaml")
        }
    }
    return metadata

def extract_file_info(parts):
    """Extract information based on expected file name structure."""
    journal_name = parts[0]
    year = parts[1]
    volume_info = 'NA'
    number_index = parts[2]
    page_index = parts[3].split('.')[0]
    return journal_name, year, volume_info, number_index, page_index

def categorize_layout_elements(text_blocks, figure_blocks):
    """Categorize layout elements for metadata."""
    identified_elements = {
        'Images': [], 'TextBlocks': [], 'Tables': [], 'Pagination': [],
        'AuthorNames': [], 'Titles': [], 'Advertisements': [], 'HeaderOrFooters': [], 'Lists': []
    }
    for block in text_blocks:
        element = block_to_metadata(block)
        identified_elements[block.type.capitalize() + 's'].append(element)

    for block in figure_blocks:
        element = block_to_metadata(block, include_content=False)
        identified_elements[block.type.capitalize() + 's'].append(element)

    return identified_elements

def block_to_metadata(block, include_content=True):
    """Convert a block to metadata format."""
    element = {
        'BoundingBox': [block.block.x_1, block.block.y_1, block.block.x_2 - block.block.x_1, block.block.y_2 - block.block.y_1],
        'Content': block.text.strip() if include_content else "",
        'Type': block.type,
        'Score': block.score,
        'ElementID': block.id if hasattr(block, 'id') else None
    }
    return element

def save_output_and_metadata(image_path, layout, metadata, output_folder, metadata_folder):
    relative_path = os.path.relpath(image_path, project_folder)
    output_dir = os.path.join(output_folder, 'v1', os.path.dirname(relative_path))
    metadata_dir = os.path.join(metadata_folder, 'v1', os.path.dirname(relative_path))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    layout_output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_layout.json'))
    with open(layout_output_path, 'w') as f:
        layout_dict = [block.to_dict() for block in layout]
        json.dump(layout_dict, f, indent=4)

    metadata_path = os.path.join(metadata_dir, os.path.basename(image_path).replace('.jpg', '.yaml'))
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

def main(project_folder, image_folders, output_folder, metadata_folder, tesseract_path, tessdata_dir):
    ocr_agent = initialize_tesseract(tesseract_path, tessdata_dir)
    model = initialize_model(model_config_path, model_path, label_map)
    for folder in image_folders:
        folder_path = os.path.join(project_folder, folder, 'images')
        for root, _, files in os.walk(folder_path):
            total_pages = sum(1 for file in files if file.endswith('.jpg'))
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(root, file)
                    logging.info(f"Processing {image_path}")
                    process_image(image_path, model, ocr_agent, output_folder, metadata_folder, total_pages)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--project_folder', required=True, help='Base project folder path')
    parser.add_argument('--output_folder', required=True, help='Output folder for layouts')
    parser.add_argument('--metadata_folder', required=True, help='Folder to save metadata files')
    parser.add_argument('--tesseract_path', required=True, help='Path to the Tesseract executable')
    parser.add_argument('--tessdata_dir', required=True, help='Path to Tesseract data directory')
    args = parser.parse_args()

    main(args.project_folder, image_folders, args.output_folder, args.metadata_folder, args.tesseract_path, args.tessdata_dir)
