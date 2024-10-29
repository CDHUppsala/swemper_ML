import os
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from argparse import ArgumentParser

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet-18 model and modify the output layer
def initialize_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()  # Modify the model to output feature vectors
    return model.to(device).eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_feature(model, image_path):
    """Extract feature vector for a single image."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image)
    return feature.cpu().numpy()

def load_dataset(directories, model):
    """Load dataset and extract features from each image in the directories."""
    all_samples = []
    all_features = []

    for dir_path in tqdm(directories, desc="Loading dataset"):
        dataset = ImageFolder(root=dir_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
        features, samples = extract_features(dataloader, model)
        all_features.append(features)
        all_samples.extend(samples)

    all_features = np.vstack(all_features)
    return all_features, all_samples

def extract_features(dataloader, model):
    """Extract features for all images in the dataloader."""
    features = []
    filenames = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            batch_filenames = [dataloader.dataset.samples[idx][0] for idx in dataloader.batch_sampler.indices]
            filenames.extend(batch_filenames)
    return np.vstack(features), filenames

def find_similar_images(source_features, dataset_features, dataset_filenames, top_k=2):
    """Compute cosine similarity scores and find top-k similar images."""
    similarity_scores = cosine_similarity(source_features, dataset_features)
    similar_images = {}
    for idx, scores in enumerate(similarity_scores):
        sorted_indices = np.argsort(scores)[::-1][1:top_k+1]  # Exclude the image itself
        similar_images[idx] = [(dataset_filenames[i], scores[i]) for i in sorted_indices]
    return similar_images

def save_results(similar_images_info, output_folder, output_csv):
    """Move similar images to a specified folder and log details."""
    os.makedirs(output_folder, exist_ok=True)
    log_details = []

    for source_idx, sims in similar_images_info.items():
        for sim_path, score in sims:
            if score < 1.0:  # Avoid exact matches
                shutil.move(sim_path, output_folder)
                log_details.append({'source': source_idx, 'similar': sim_path, 'score': score})

    # Save log details to CSV
    pd.DataFrame(log_details).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def main(source_file_paths, dataset_directories, output_folder, output_csv):
    # Initialize model
    model = initialize_model()

    # Load dataset
    dataset_features, dataset_filenames = load_dataset(dataset_directories, model)

    # Load source images and compute their features
    source_features = np.array([get_image_feature(model, path) for path in source_file_paths])

    # Compare source images with the dataset
    similar_images_info = find_similar_images(source_features, dataset_features, dataset_filenames)

    # Move files and log details
    save_results(similar_images_info, output_folder, output_csv)

if __name__ == '__main__':
    parser = ArgumentParser(description="Find similar images in dataset directories")
    parser.add_argument("--source_file_paths", nargs='+', required=True, help="List of source image paths")
    parser.add_argument("--dataset_directories", nargs='+', required=True, help="Directories containing dataset images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save similar images")
    parser.add_argument("--output_csv", type=str, default="image_tracking.csv", help="CSV to save similarity results")

    args = parser.parse_args()
    main(args.source_file_paths, args.dataset_directories, args.output_folder, args.output_csv)
