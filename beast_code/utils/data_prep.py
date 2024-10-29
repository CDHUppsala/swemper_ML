import os
import random
import csv
from shutil import copy2
import argparse

def create_directory(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_csv_header(csv_path):
    """Prepare the CSV file and write the header."""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Periodical', 'Year', 'OriginalPath', 'SelectedPath'])

def select_images(src_directory, dest_directory, csv_path, target_periodicals, images_per_periodical=200):
    """Select random images from the source directory and copy them to the destination directory.

    Parameters:
        src_directory (str): Path to the source directory containing images.
        dest_directory (str): Path to the destination directory for selected images.
        csv_path (str): Path to the CSV file for logging selected images.
        target_periodicals (list): List of periodicals to target.
        images_per_periodical (int): Number of images to select per periodical.
    """
    create_directory(dest_directory)
    write_csv_header(csv_path)

    # Selecting images based on periodicals
    for periodical in target_periodicals:
        periodical_images = [img for img in os.listdir(src_directory) if periodical in img]
        selected_images = random.sample(periodical_images, min(images_per_periodical, len(periodical_images)))

        for img in selected_images:
            src_path = os.path.join(src_directory, img)
            dest_path = os.path.join(dest_directory, img)
            copy2(src_path, dest_path)  # Copy the image
            with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([periodical, "Year Placeholder", src_path, dest_path])  # Year Placeholder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select images from source directory.')
    parser.add_argument('--src_directory', type=str, required=True, help='Path to the source directory containing images.')
    parser.add_argument('--dest_directory', type=str, required=True, help='Path to the destination directory for selected images.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file for logging selected images.')
    parser.add_argument('--target_periodicals', type=str, nargs='+', required=True, help='List of periodicals to target.')
    parser.add_argument('--images_per_periodical', type=int, default=200, help='Number of images to select per periodical.')

    args = parser.parse_args()

    select_images(args.src_directory, args.dest_directory, args.csv_path, args.target_periodicals, args.images_per_periodical)
