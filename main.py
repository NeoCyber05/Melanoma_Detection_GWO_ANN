"""
Main - Phân loại Melanoma với ANN và GWO-ANN
"""

import numpy as np
import torch
import random

from data_loader import get_data_loaders
from ann_model import MelanomaANN
from gwo_ann_trainer import HybridGWOANNTrainer, StandardANNTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # === CONFIG ===
    DATA_DIR = 'melanoma_cancer_dataset'
    IMG_SIZE = 64
    PCA_COMPONENTS = 100
    HIDDEN_DIM = 64
    
    GWO_POPULATION = 30
    GWO_ITERATIONS = 50
    
    BP_EPOCHS = 100
    BP_BATCH = 32
    BP_LR = 0.001
    PATIENCE = 15
    
    # === LOAD DATA ===
    print("\n[1] Loading data...")
    (X_train, y_train), (X_test, y_test), input_dim = get_data_loaders(
        DATA_DIR, IMG_SIZE, use_pca=True, n_components=PCA_COMPONENTS
    )
    
    # Split train/val
    val_size = int(0.2 * len(X_train))
    idx = np.random.permutation(len(X_train))
    X_val, y_val = X_train[idx[:val_size]], y_train[idx[:val_size]]
    X_train, y_train = X_train[idx[val_size:]], y_train[idx[val_size:]]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # =============================================
    # STANDARD ANN (chỉ BP)
    # =============================================
    print("\n" + "="*60)
    print("TRAINING STANDARD ANN (Baseline)")
    print("="*60)
    
    set_seed(42)
    model_std = MelanomaANN(input_dim, HIDDEN_DIM)
    trainer_std = StandardANNTrainer(model_std, device)
    trainer_std.train(X_train, y_train, X_val, y_val, 
                      epochs=BP_EPOCHS, batch_size=BP_BATCH, lr=BP_LR, patience=PATIENCE)
    results_std = trainer_std.evaluate(X_test, y_test)
    trainer_std.save_model('model_standard_ann.pth')
    
    # =============================================
    # HYBRID GWO-ANN
    # =============================================
    print("\n" + "="*60)
    print("TRAINING HYBRID GWO-ANN")
    print("="*60)
    
    set_seed(42)
    model_gwo = MelanomaANN(input_dim, HIDDEN_DIM)
    trainer_gwo = HybridGWOANNTrainer(model_gwo, device)
    trainer_gwo.train(X_train, y_train, X_val, y_val,
                      gwo_pop=GWO_POPULATION, gwo_iter=GWO_ITERATIONS,
                      bp_epochs=BP_EPOCHS, bp_batch=BP_BATCH, bp_lr=BP_LR, patience=PATIENCE)
    results_gwo = trainer_gwo.evaluate(X_test, y_test)
    trainer_gwo.save_model('model_gwo_ann.pth')
    
    # =============================================
    # SO SÁNH KẾT QUẢ
    # =============================================
    print("\n" + "="*60)
    print("COMPARISON: STANDARD ANN vs GWO-ANN")
    print("="*60)
    
    print(f"{'Metric':<15} {'Standard ANN':>15} {'GWO-ANN':>15} {'Improvement':>15}")
    print("-" * 60)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']:
        std_val = results_std[metric] * 100
        gwo_val = results_gwo[metric] * 100
        diff = gwo_val - std_val
        print(f"{metric:<15} {std_val:>14.2f}% {gwo_val:>14.2f}% {diff:>+14.2f}%")
    
    print("\nDone!")


def train_standard_only():
    """Chỉ train Standard ANN"""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    (X_train, y_train), (X_test, y_test), input_dim = get_data_loaders(
        'melanoma_cancer_dataset', img_size=64, use_pca=True, n_components=100
    )
    
    val_size = int(0.2 * len(X_train))
    idx = np.random.permutation(len(X_train))
    X_val, y_val = X_train[idx[:val_size]], y_train[idx[:val_size]]
    X_train, y_train = X_train[idx[val_size:]], y_train[idx[val_size:]]
    
    model = MelanomaANN(input_dim, 64)
    trainer = StandardANNTrainer(model, device)
    trainer.train(X_train, y_train, X_val, y_val, epochs=100, patience=15)
    trainer.evaluate(X_test, y_test)
    trainer.save_model('model_standard_ann.pth')


def train_gwo_only():
    """Chỉ train GWO-ANN"""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    (X_train, y_train), (X_test, y_test), input_dim = get_data_loaders(
        'melanoma_cancer_dataset', img_size=64, use_pca=True, n_components=100
    )
    
    val_size = int(0.2 * len(X_train))
    idx = np.random.permutation(len(X_train))
    X_val, y_val = X_train[idx[:val_size]], y_train[idx[:val_size]]
    X_train, y_train = X_train[idx[val_size:]], y_train[idx[val_size:]]
    
    model = MelanomaANN(input_dim, 64)
    trainer = HybridGWOANNTrainer(model, device)
    trainer.train(X_train, y_train, X_val, y_val, gwo_pop=30, gwo_iter=50, bp_epochs=100, patience=15)
    trainer.evaluate(X_test, y_test)
    trainer.save_model('model_gwo_ann.pth')


def quick_test():
    """Test nhanh với data nhỏ"""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Quick test...")
    (X_train, y_train), (X_test, y_test), dim = get_data_loaders(
        'melanoma_cancer_dataset', img_size=32, use_pca=True, n_components=50
    )
    
    X_train, y_train = X_train[:500], y_train[:500]
    X_test, y_test = X_test[:100], y_test[:100]
    
    # Standard
    print("\n--- Standard ANN ---")
    model1 = MelanomaANN(dim, 32)
    t1 = StandardANNTrainer(model1, device)
    t1.train(X_train, y_train, epochs=20)
    r1 = t1.evaluate(X_test, y_test)
    
    # GWO-ANN
    print("\n--- GWO-ANN ---")
    model2 = MelanomaANN(dim, 32)
    t2 = HybridGWOANNTrainer(model2, device)
    t2.train(X_train, y_train, gwo_pop=10, gwo_iter=10, bp_epochs=20)
    r2 = t2.evaluate(X_test, y_test)
    
    print(f"\nStandard: {r1['accuracy']*100:.2f}% | GWO-ANN: {r2['accuracy']*100:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            quick_test()
        elif sys.argv[1] == '--standard':
            train_standard_only()
        elif sys.argv[1] == '--gwo':
            train_gwo_only()
        else:
            print("Options: --quick, --standard, --gwo, or no args for full comparison")
    else:
        main()
