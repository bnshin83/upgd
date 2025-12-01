#!/usr/bin/env python3
"""
Download and prepare Mini-ImageNet dataset for UPGD experiments.

The full Mini-ImageNet dataset has:
- 100 classes
- 600 images per class = 60,000 total images
- Images are 84x84x3

This script will:
1. Download Mini-ImageNet using the datasets library
2. Save as mini-imagenet_data.pkl and mini-imagenet_targets.pkl
3. Optionally process through ResNet50 to create processed_imagenet.pkl

Download from: Hugging Face (timm/mini-imagenet)
Cache to: /scratch/gilbreth/shin283/.cache/huggingface/
Save to: /scratch/gilbreth/shin283/upgd/dataset/
"""

import os
import pickle
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

# Set Hugging Face cache to scratch (not home folder)
SCRATCH_CACHE = "/scratch/gilbreth/shin283/.cache/huggingface"
os.makedirs(SCRATCH_CACHE, exist_ok=True)
os.environ["HF_HOME"] = SCRATCH_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(SCRATCH_CACHE, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(SCRATCH_CACHE, "transformers")

def download_mini_imagenet_huggingface():
    """Download Mini-ImageNet from Hugging Face."""
    try:
        from datasets import load_dataset
        print("Downloading Mini-ImageNet from Hugging Face...")
        dataset = load_dataset("timm/mini-imagenet")
        return dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return None

def download_mini_imagenet_mlclf():
    """Download Mini-ImageNet using MLclf package."""
    try:
        from MLclf import MLclf
        print("Downloading Mini-ImageNet using MLclf...")
        MLclf.miniimagenet_download(Download=True)
        train_dataset, val_dataset, test_dataset = MLclf.miniimagenet_clf_dataset(
            ratio_train=0.6, ratio_val=0.2, seed_value=42, shuffle=True
        )
        return train_dataset, val_dataset, test_dataset
    except ImportError:
        print("Please install MLclf: pip install MLclf")
        return None

def prepare_dataset_from_huggingface(dataset, output_dir="dataset"):
    """Convert Hugging Face dataset to pickle format expected by UPGD."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine train and validation splits
    all_images = []
    all_labels = []
    
    for split_name in ['train', 'validation']:
        if split_name in dataset:
            split = dataset[split_name]
            print(f"Processing {split_name} split: {len(split)} samples")
            
            for item in tqdm(split, desc=f"Processing {split_name}"):
                # Get image and resize to 84x84
                img = item['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((84, 84), Image.BILINEAR)
                img_array = np.array(img)
                all_images.append(img_array)
                all_labels.append(item['label'])
    
    # Convert to numpy arrays
    data = np.array(all_images, dtype=np.uint8)
    targets = np.array(all_labels)
    
    print(f"\nTotal samples: {len(data)}")
    print(f"Data shape: {data.shape}")
    print(f"Unique classes: {len(np.unique(targets))}")
    
    # Save as pickle files
    data_path = os.path.join(output_dir, "mini-imagenet_data.pkl")
    targets_path = os.path.join(output_dir, "mini-imagenet_targets.pkl")
    
    print(f"\nSaving data to {data_path}...")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saving targets to {targets_path}...")
    with open(targets_path, 'wb') as f:
        pickle.dump(targets, f)
    
    print("Done!")
    return data, targets

def get_bottle_neck(model, x):
    """Extract ResNet50 bottleneck features."""
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)

def process_with_resnet50(data, output_file="processed_imagenet.pkl", batch_size=100):
    """Process images through ResNet50 to get bottleneck features."""
    print("\nProcessing images through ResNet50...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained ResNet50
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet = resnet.to(device)
    resnet.eval()
    
    for param in resnet.parameters():
        param.requires_grad_(False)
    
    # Transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Process in batches
    n_samples = len(data)
    processed_data = torch.zeros((n_samples, resnet.fc.in_features))
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, n_samples)
            batch_images = []
            
            for j in range(i, batch_end):
                img = Image.fromarray(data[j])
                img_tensor = transform(img)
                batch_images.append(img_tensor)
            
            batch_tensor = torch.stack(batch_images).to(device)
            features = get_bottle_neck(resnet, batch_tensor)
            processed_data[i:batch_end] = features.cpu()
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    
    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Done!")
    return processed_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and prepare Mini-ImageNet dataset")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for dataset files")
    parser.add_argument("--skip-resnet", action="store_true", help="Skip ResNet50 processing")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for ResNet50 processing")
    args = parser.parse_args()
    
    # Try Hugging Face first
    dataset = download_mini_imagenet_huggingface()
    
    if dataset is not None:
        data, targets = prepare_dataset_from_huggingface(dataset, args.output_dir)
        
        if not args.skip_resnet:
            process_with_resnet50(data, batch_size=args.batch_size)
    else:
        print("\nFailed to download dataset. Please install required packages:")
        print("  pip install datasets")
        print("  # or")
        print("  pip install MLclf")

if __name__ == "__main__":
    main()

