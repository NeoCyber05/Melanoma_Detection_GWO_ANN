"""
Data Loader cho Melanoma dataset
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm


class MelanomaDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load benign (0) và malignant (1)
        for label, folder in enumerate(['benign', 'malignant']):
            folder_path = os.path.join(root_dir, folder)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(folder_path, img_name))
                        self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images: {self.labels.count(0)} benign, {self.labels.count(1)} malignant")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_transforms(img_size=64):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def extract_features(dataset):
    """Trích xuất features từ ảnh"""
    features, labels = [], []
    
    print("Extracting features...")
    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        features.append(img.numpy().flatten())
        labels.append(label)
    
    return np.array(features), np.array(labels)


def get_data_loaders(data_dir, img_size=64, use_pca=True, n_components=100):
    """Load data và trả về features"""
    transform = get_transforms(img_size)
    
    train_dataset = MelanomaDataset(os.path.join(data_dir, 'train'), transform)
    test_dataset = MelanomaDataset(os.path.join(data_dir, 'test'), transform)
    
    train_features, train_labels = extract_features(train_dataset)
    test_features, test_labels = extract_features(test_dataset)
    
    # PCA
    if use_pca:
        print(f"Applying PCA: {train_features.shape[1]} -> {n_components}")
        pca = PCA(n_components=n_components)
        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        print(f"Explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Normalize
    mean, std = train_features.mean(0), train_features.std(0) + 1e-8
    train_features = (train_features - mean) / std
    test_features = (test_features - mean) / std
    
    input_dim = train_features.shape[1]
    print(f"Final dim: {input_dim}")
    
    return (train_features, train_labels), (test_features, test_labels), input_dim


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), dim = get_data_loaders(
        "melanoma_cancer_dataset", img_size=64, use_pca=True, n_components=100
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
