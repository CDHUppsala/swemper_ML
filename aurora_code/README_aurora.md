
# swemper_ML Project

## Overview
This project includes a machine learning pipeline for layout detection using Mask R-CNN, with tools for data preparation and model training. The setup includes scripts to handle data processing and training configuration for easy deployment.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/swemper_ML.git
   cd swemper_ML
   ```

2. **Install Dependencies**:
   Ensure you have the required dependencies installed:
   ```bash
   pip install -r aurora/requirements.txt
   ```

3. **Install Layout Parser Toolkit**:
   To install the Layout Parser toolkit, run the following command:
   ```bash
   pip install layoutparser #refer to the installation instructions at "https://github.com/Layout-Parser/layout-parser" if you run into issues related to layout parser
   ```

## Usage

### 1. Data Preparation

Use the `data_handling.py` script to prepare the dataset. You can specify the dataset directory and the zip file as arguments, or download directly from Roboflow.

**To download from Roboflow:**
```bash
python aurora/data_handling.py --dataset_path "./Training_WS.v4-downscaled-800x1200" --api_key "YOUR_ROBOFLOW_API_KEY" --workspace "YOUR_WORKSPACE_NAME" --project "YOUR_PROJECT_NAME" --version 1
```

**To use a local zip file:**
```bash
python aurora/data_handling.py --dataset_path "./Training_WS.v4-downscaled-800x1200" --zip_path "Training_WS.v4-downscaled-800x1200-no_validation.coco.zip"
```

### 2. Model Training

To train the model, use `train_net.py` with your configuration file.

   ```bash
   python aurora/tools/train_net.py --dataset_name "swemper_dataset" \
       --json_annotation_train "train/_annotations.coco.json" \
       --image_path_train "train" \
       --json_annotation_val "test/_annotations.coco.json" \
       --image_path_val "test" \
       --config-file "aurora/config_mask_rcnn_resized.yaml" \
       OUTPUT_DIR "outputs" MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE 512 \
       SOLVER.CHECKPOINT_PERIOD 500 SOLVER.MAX_ITER 4000 SOLVER.IMS_PER_BATCH 2
   ```

Replace paths and parameters as needed to customize training.

## License Information

This project uses the following libraries:

- **Layout Parser**: Licensed under the [MIT License](https://github.com/Layout-Parser/layout-parser/blob/main/LICENSE). Please refer to the license for details on usage and distribution.

## Directory Structure

- `common_code/`: Shared code and utilities
- `aurora/`: Specific files and scripts for Machine "Aurora". Houses code for training
- `beast/`: Specific files and scripts for Machine "Beast". Houses code for data prep for inference, inference and create metadata

