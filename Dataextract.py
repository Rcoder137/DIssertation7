import zipfile
import os

# Path to the zip file
zip_path = r'C:\Users\robin\Projects\CropClassificationProject\Datasets\Archive2.zip'
extract_folder = r'C:\Users\robin\Projects\CropClassificationProject\Datasets\crop_dataset'

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# List contents of the extracted folder
extracted_files = os.listdir(extract_folder)
extracted_files
