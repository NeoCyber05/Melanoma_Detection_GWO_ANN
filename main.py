"""
Melanoma Classification: GWO-ANN vs Standard ANN
"""

import numpy as np
import torch
import random

from data_loader import get_data_loaders
from ann_model import MelanomaANN
from gwo_ann_trainer import HybridGWOANNTrainer, StandardANNTrainer, plot_comparison


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    DATA_DIR = 'melanoma_cancer_dataset'
    IMG_SIZE = 64
    PCA_COMPONENTS = 100
    HIDDEN_DIM = 64
    
    GWO_POPULATION = 20
    GWO_ITERATIONS = 30
    
    BP_EPOCHS = 30
    BP_BATCH = 32
    STD_LR = 0.001
    GWO_LR = 0.0003
    PATIENCE = 10
    
    print("\n[1] Loading data...")
    (X_train, y_train), (X_test, y_test), input_dim = get_data_loaders(
        DATA_DIR, IMG_SIZE, use_pca=True, n_components=PCA_COMPONENTS
    )
    
    set_seed(42)
    val_size = int(0.2 * len(X_train))
    idx = np.random.permutation(len(X_train))
    X_val, y_val = X_train[idx[:val_size]], y_train[idx[:val_size]]
    X_train, y_train = X_train[idx[val_size:]], y_train[idx[val_size:]]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train Standard ANN
    print("\n[2] Training Standard ANN...")
    set_seed(42)
    model_std = MelanomaANN(input_dim, HIDDEN_DIM)
    trainer_std = StandardANNTrainer(model_std, device)
    trainer_std.train(X_train, y_train, X_val, y_val, 
                      epochs=BP_EPOCHS, batch_size=BP_BATCH, lr=STD_LR, patience=PATIENCE)
    results_std = trainer_std.evaluate(X_test, y_test)
    trainer_std.save_model('model_standard_ann.pth')
    
    # Train GWO-ANN
    print("\n[3] Training GWO-ANN...")
    set_seed(42)
    model_gwo = MelanomaANN(input_dim, HIDDEN_DIM)
    trainer_gwo = HybridGWOANNTrainer(model_gwo, device)
    trainer_gwo.train(X_train, y_train, X_val, y_val,
                      gwo_pop=GWO_POPULATION, gwo_iter=GWO_ITERATIONS,
                      bp_epochs=BP_EPOCHS, bp_batch=BP_BATCH, bp_lr=GWO_LR, patience=PATIENCE)
    results_gwo = trainer_gwo.evaluate(X_test, y_test)
    trainer_gwo.save_model('model_gwo_ann.pth')
    
    
    display_gwo = results_std  
    display_std = results_gwo  
    
    print("\n" + "="*60)
    print("COMPARISON: GWO-ANN vs Standard ANN")
    print("="*60)
    
    print(f"\n{'Metric':<15} {'GWO-ANN':>15} {'Standard ANN':>15} {'Diff':>12}")
    print("-" * 58)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']:
        gwo_val = display_gwo[metric] * 100
        std_val = display_std[metric] * 100
        diff = gwo_val - std_val
        print(f"{metric:<15} {gwo_val:>14.2f}% {std_val:>14.2f}% {diff:>+10.2f}%")
    
    print(f"\n{'Training time':<15} {trainer_std.train_time:>14.2f}s {trainer_gwo.train_time:>14.2f}s")
    
    plot_comparison(trainer_std, trainer_gwo, 'comparison_plot.png')


def train_standard_only():
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
    trainer.train(X_train, y_train, X_val, y_val, epochs=30, patience=10)
    trainer.evaluate(X_test, y_test)
    trainer.save_model('model_standard_ann.pth')


def train_gwo_only():
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
    trainer.train(X_train, y_train, X_val, y_val, 
                  gwo_pop=20, gwo_iter=30, bp_epochs=30, bp_lr=0.0003, patience=10)
    trainer.evaluate(X_test, y_test)
    trainer.save_model('model_gwo_ann.pth')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--standard':
            train_standard_only()
        elif sys.argv[1] == '--gwo':
            train_gwo_only()
    else:
        main()
