import os
import subprocess
import shutil
import random
import urllib.request
import zipfile
import kagglehub
from pathlib import Path

def run_cmd(cmd):
    print(f"\n======================================")
    print(f"Running: {' '.join(cmd)}")
    print(f"======================================")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        exit(1)

def prepare_plantvillage_dataset():
    print("\n======================================")
    print("Preparing Full PlantVillage Dataset (38 Classes)")
    print("======================================")
    
    # Check if we already have the full 38 classes
    if os.path.exists("PlantVillage") and len(os.listdir("PlantVillage")) >= 38:
        print("PlantVillage dataset already has 38 classes. Skipping download.")
        return

    # Delete the incomplete 15-class dataset if it exists
    if os.path.exists("PlantVillage"):
        print("Removing incomplete PlantVillage folder...")
        shutil.rmtree("PlantVillage")

    print("Downloading full dataset from Kaggle...")
    path = kagglehub.dataset_download('abdallahalidev/plantvillage-dataset')
    print(f"Downloaded to {path}")

    # The dataset usually has structure: plantvillage dataset/color/
    # We need to find the 'color' folder which contains the 38 classes
    source_dir = None
    for root, dirs, files in os.walk(path):
        if 'color' in dirs and len(os.listdir(os.path.join(root, 'color'))) >= 38:
            source_dir = os.path.join(root, 'color')
            break
        elif len(dirs) >= 38:
            source_dir = root
            break

    if not source_dir:
        print("ERROR: Could not locate the 38-class folder in the downloaded dataset.")
        exit(1)

    print(f"Found 38-class dataset at: {source_dir}")
    print("Copying to ./PlantVillage ...")
    shutil.copytree(source_dir, "PlantVillage", dirs_exist_ok=True)
    print("PlantVillage dataset prepared successfully.")

def prepare_validator_data():
    print("\n======================================")
    print("Preparing Validator Data")
    print("======================================")
    os.makedirs('data/validator/plant', exist_ok=True)
    os.makedirs('data/validator/nonplant', exist_ok=True)

    print("Copying 10,000 plant images from PlantVillage...")
    all_imgs = list(Path('PlantVillage').rglob('*.jpg')) + list(Path('PlantVillage').rglob('*.JPG'))
    
    # Clear existing plant images if we restart
    for f in Path('data/validator/plant').glob('*.jpg'):
        f.unlink()

    sampled = random.sample(all_imgs, min(10000, len(all_imgs)))
    for i, p in enumerate(sampled):
        shutil.copy(p, f'data/validator/plant/plant_{i:05d}.jpg')

    # Download nonplant if not exists or if empty
    if len(list(Path('data/validator/nonplant').glob('*.jpg'))) < 100:
        print("Downloading ImageNet-mini nonplant subset via kagglehub...")
        path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
        
        print(f"Downloaded to {path}. Copying 5000 nonplant images...")
        nonplant_imgs = list(Path(path).rglob('*.JPEG')) + list(Path(path).rglob('*.jpg'))
        sampled_nonplant = random.sample(nonplant_imgs, min(5000, len(nonplant_imgs)))
        for i, p in enumerate(sampled_nonplant):
            shutil.copy(p, f'data/validator/nonplant/nonplant_{i:05d}.jpg')
    else:
        print("Nonplant dataset already exists.")

    print("Validator data prepared successfully.")

if __name__ == "__main__":
    print("Starting full training pipeline...")

    # 1. Prepare Dataset (Fixing the 15-class issue)
    # prepare_plantvillage_dataset() # Skipped, already done

    # 2. Train Classifier
    # run_cmd(["python", "training/train_plant_classifier.py"]) # Skipped, already done

    # 3. Prepare Validator Data
    # prepare_validator_data()

    # 4. Train Validator
    run_cmd(["python", "training/train_validator.py"])

    # 5. Build RAG Index
    run_cmd(["python", "-c", "from services.rag_faiss import rag_pipeline; rag_pipeline.build_index()"])

    print("\n✅ All training completed successfully!")
