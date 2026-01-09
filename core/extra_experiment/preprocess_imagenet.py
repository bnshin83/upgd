#!/usr/bin/env python
"""
Pre-process Mini-ImageNet through ResNet-50 to extract bottleneck features.
This creates processed_imagenet.pkl with 2048-dim feature vectors for each image.
"""
import torch
import torchvision
from PIL import Image
import pickle
import time

def get_bottle_neck(model, x):
    """Extract bottleneck features from ResNet-50 (before FC layer)."""
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

def main():
    print("=" * 60)
    print("Mini-ImageNet ResNet-50 Feature Extraction")
    print("=" * 60)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load raw images
    print("\nLoading raw images from dataset/mini-imagenet_data.pkl...")
    start_time = time.time()
    with open('dataset/mini-imagenet_data.pkl', 'rb') as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} images in {time.time() - start_time:.1f}s")
    print(f"Image shape: {raw_data[0].shape}")
    
    # Transform images
    print("\nTransforming images...")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,), (0.5,)),
    ])
    
    start_time = time.time()
    images = [transform(Image.fromarray(img)) for img in raw_data]
    data = torch.stack(images)
    print(f"Transformed to tensor shape: {data.shape} in {time.time() - start_time:.1f}s")
    
    # Load ResNet-50
    print("\nLoading ResNet-50 pretrained model...")
    resnet = torchvision.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad_(False)
    resnet.eval()
    resnet = resnet.to(device)
    print("ResNet-50 loaded and moved to device")
    
    # Process through ResNet-50
    print("\nExtracting bottleneck features...")
    n_samples = data.shape[0]
    processed_data = torch.zeros((n_samples, resnet.fc.in_features))
    batch_size = 256 if device.type == 'cuda' else 100
    
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = data[i:end_idx].to(device)
            features = get_bottle_neck(resnet, batch)
            processed_data[i:end_idx] = features.cpu()
            
            # Progress update
            progress = (end_idx / n_samples) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / end_idx) * (n_samples - end_idx) if end_idx > 0 else 0
            print(f"  Processed {end_idx:,}/{n_samples:,} ({progress:.1f}%) - ETA: {eta:.0f}s")
    
    total_time = time.time() - start_time
    print(f"\nFeature extraction completed in {total_time:.1f}s")
    print(f"Output shape: {processed_data.shape}")
    
    # Save processed features
    print("\nSaving to processed_imagenet.pkl...")
    with open('processed_imagenet.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    # Verify file size
    import os
    file_size = os.path.getsize('processed_imagenet.pkl') / (1024 * 1024)
    print(f"Saved! File size: {file_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("SUCCESS! processed_imagenet.pkl is ready for experiments")
    print("=" * 60)

if __name__ == "__main__":
    main()

