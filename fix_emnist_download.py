#!/usr/bin/env python3
"""Fix EMNIST dataset download issue"""

import os
import torchvision

# Create dataset directory
os.makedirs('dataset', exist_ok=True)

# Download EMNIST dataset with error handling
try:
    print("Downloading EMNIST dataset...")
    dataset = torchvision.datasets.EMNIST(
        root='dataset',
        split='byclass',
        train=True,
        download=True
    )
    print(f"✓ EMNIST dataset downloaded successfully: {len(dataset)} training samples")
    
    # Also download test set
    test_dataset = torchvision.datasets.EMNIST(
        root='dataset',
        split='byclass',
        train=False,
        download=True
    )
    print(f"✓ EMNIST test set downloaded: {len(test_dataset)} test samples")
    
except Exception as e:
    print(f"✗ Failed to download EMNIST: {e}")
    print("Trying alternative download method...")
    # Alternative: manual download instructions
    print("""
    Manual download instructions:
    1. Download from: https://www.nist.gov/itl/products-and-services/emnist-dataset
    2. Extract to: dataset/EMNIST/
    """)
