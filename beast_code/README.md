
# Swemper Machine Learning: Beast code

This repository contains a set of Python scripts for data preparation, segmentation, image similarity search, and metadata creation. These scripts are on Beast for the Swemper project.

## Requirements

Please refer to `requirements.txt` for package dependencies. **Note**: This project uses Tesseract OCR and Layout Parser's Layout Detection Model, requiring additional setup for effective use.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/CDHUppsala/swemper_ML.git
   cd swemper_ML/beast
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - Ensure Tesseract is installed and accessible from the command line. Instructions are available on [Tesseract OCR's GitHub page](https://github.com/tesseract-ocr/tesseract).
   - Add the path to `tessdata` in your `TESSDATA_PREFIX` environment variable if needed.

### Usage

Each script has specific command-line arguments. Below is a summary of their functionalities and usage instructions.

#### 1. `data_prep.py`

This script selects images from the source directory and saves them to a specified destination. Additionally, it logs details of the selected images in a CSV.

**Usage**:
```bash
python data_prep.py --src_directory /path/to/src --dest_directory /path/to/dest --csv_path /path/to/log.csv --target_periodicals Periodical1 Periodical2 --images_per_periodical 200
```

#### 2. `add_segmentation_data.py`

Adds segmentation data to image annotations.

**Usage**:
```bash
python add_segmentation_data.py --source_file /path/to/annotations.json --output_file /path/to/updated_annotations.json
```

#### 3. `find_similar_images.py`

Finds images similar to a set of source images within a dataset directory. Similar images are moved to a specified output folder, and results are logged in a CSV file.

**Usage**:
```bash
python find_similar_images.py --source_file_paths /path/to/source1.jpg /path/to/source2.jpg --dataset_directories /path/to/dataset1 /path/to/dataset2 --output_folder /path/to/similar_images --output_csv /path/to/similarity_results.csv
```

#### 4. `create_metadata_v1.py`

Generates metadata for images, using Tesseract for OCR and Layout Parser for layout detection.

**Usage**:
```bash
python create_metadata_v1.py --project_folder /path/to/project --output_folder /path/to/output --metadata_folder /path/to/metadata --tesseract_path /usr/local/bin/tesseract --tessdata_dir /usr/local/share/tessdata
```



