"""
ANN Trainers: Standard (BP) và Hybrid (GWO + BP)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import matplotlib.pyplot as plt

from ann_model import MelanomaANN, calculate_accuracy
from gwo_optimizer import GreyWolfOptimizer, create_objective_function


class StandardANNTrainer:
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model._init_weights()
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.gwo_history = []
        self.bp_history = self.history
        self.train_time = 0
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, lr=0.001, patience=10, verbose=True):
        
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)
        
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        wait = 0
        best_state = None
        
        start = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            loss_sum, correct, total = 0, 0, 0
            
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
                loss_sum += loss.item()
                correct += (out.argmax(1) == by).sum().item()
                total += by.size(0)
            
            train_loss = loss_sum / len(loader)
            train_acc = correct / total
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            if X_val is not None:
                val_loss, val_acc = self._eval(X_val, y_val, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    best_state = self.model.state_dict().copy()
                else:
                    wait += 1
                
                if wait >= patience:
                    self.model.load_state_dict(best_state)
                    break
                
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val: {val_loss:.4f} {val_acc*100:.1f}%")
        
        self.train_time = time.time() - start
    
    def _eval(self, X, y, criterion):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            loss = criterion(out, y_t).item()
            acc = (out.argmax(1) == y_t).float().mean().item()
        return loss, acc
    
    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_t = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            preds = self.model.predict(X_t).cpu().numpy()
        
        y_true = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, average='weighted')
        rec = recall_score(y_true, preds, average='weighted')
        f1 = f1_score(y_true, preds, average='weighted')
        cm = confusion_matrix(y_true, preds)
        
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        print(f"Standard ANN - Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'sensitivity': sens, 'specificity': spec, 'cm': cm}
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class HybridGWOANNTrainer:
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.gwo_history = []
        self.bp_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.train_time = 0
    
    def gwo_phase(self, X_train, y_train, X_val=None, y_val=None,
                  population_size=30, max_iter=50, verbose=True):
        
        objective = create_objective_function(
            self.model, X_train, y_train, X_val, y_val, self.device
        )
        
        dim = self.model.get_num_params()
        lb, ub = self.model.get_weight_bounds()
        
        gwo = GreyWolfOptimizer(objective, dim, population_size, max_iter, lb, ub, verbose)
        
        start = time.time()
        best_weights, best_mse = gwo.optimize()
        
        self.gwo_history = gwo.get_convergence_curve().tolist()
        self.model.set_weights_from_vector(best_weights)
        
        acc = calculate_accuracy(self.model, X_train, y_train, self.device)
        print(f"GWO done in {time.time()-start:.2f}s | MSE: {best_mse:.6f} | Acc: {acc*100:.2f}%")
        
        return best_weights, best_mse
    
    def bp_phase(self, X_train, y_train, X_val=None, y_val=None,
                 epochs=100, batch_size=32, lr=0.0005, patience=10, verbose=True):
        
        X_t = torch.FloatTensor(X_train).to(self.device)
        y_t = torch.LongTensor(y_train).to(self.device)
        
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        wait = 0
        best_state = self.model.state_dict().copy()
        
        start = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            loss_sum, correct, total = 0, 0, 0
            
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
                loss_sum += loss.item()
                correct += (out.argmax(1) == by).sum().item()
                total += by.size(0)
            
            train_loss = loss_sum / len(loader)
            train_acc = correct / total
            self.bp_history['train_loss'].append(train_loss)
            self.bp_history['train_acc'].append(train_acc)
            
            if X_val is not None:
                val_loss, val_acc = self._eval(X_val, y_val, criterion)
                self.bp_history['val_loss'].append(val_loss)
                self.bp_history['val_acc'].append(val_acc)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                    best_state = self.model.state_dict().copy()
                else:
                    wait += 1
                
                if wait >= patience:
                    self.model.load_state_dict(best_state)
                    break
                
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val: {val_loss:.4f} {val_acc*100:.1f}%")
    
    def _eval(self, X, y, criterion):
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        with torch.no_grad():
            out = self.model(X_t)
            loss = criterion(out, y_t).item()
            acc = (out.argmax(1) == y_t).float().mean().item()
        return loss, acc
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              gwo_pop=30, gwo_iter=50, bp_epochs=100, bp_batch=32, bp_lr=0.0005, patience=10):
        
        start = time.time()
        self.gwo_phase(X_train, y_train, X_val, y_val, gwo_pop, gwo_iter)
        self.bp_phase(X_train, y_train, X_val, y_val, bp_epochs, bp_batch, bp_lr, patience)
        self.train_time = time.time() - start
    
    def evaluate(self, X_test, y_test):
        self.model.eval()
        X_t = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            preds = self.model.predict(X_t).cpu().numpy()
        
        y_true = y_test if isinstance(y_test, np.ndarray) else y_test.numpy()
        
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, average='weighted')
        rec = recall_score(y_true, preds, average='weighted')
        f1 = f1_score(y_true, preds, average='weighted')
        cm = confusion_matrix(y_true, preds)
        
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        print(f"GWO-ANN - Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'sensitivity': sens, 'specificity': spec, 'cm': cm}
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


def plot_comparison(gwo_trainer, std_trainer, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Row 1: GWO-ANN
    ax1 = axes[0, 0]
    gwo_len = len(gwo_trainer.gwo_history) if gwo_trainer.gwo_history else 0
    if gwo_len > 0:
        ax1.plot(range(gwo_len), gwo_trainer.gwo_history, 'g-', label='GWO Phase')
        ax1.axvline(x=gwo_len, color='red', linestyle='--', alpha=0.5)
    if gwo_trainer.bp_history['train_loss']:
        bp_len = len(gwo_trainer.bp_history['train_loss'])
        x_bp = range(gwo_len, gwo_len + bp_len)
        ax1.plot(x_bp, gwo_trainer.bp_history['train_loss'], 'b-', label='Train')
        if gwo_trainer.bp_history['val_loss']:
            ax1.plot(x_bp, gwo_trainer.bp_history['val_loss'], 'orange', label='Val')
    ax1.set_title('GWO-ANN - Loss')
    ax1.set_xlabel('Iteration/Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    axes[0, 1].plot([a*100 for a in gwo_trainer.bp_history['train_acc']], label='Train')
    if gwo_trainer.bp_history['val_acc']:
        axes[0, 1].plot([a*100 for a in gwo_trainer.bp_history['val_acc']], label='Val')
    axes[0, 1].set_title('GWO-ANN - Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Standard ANN
    axes[1, 0].plot(std_trainer.bp_history['train_loss'], label='Train')
    if std_trainer.bp_history['val_loss']:
        axes[1, 0].plot(std_trainer.bp_history['val_loss'], label='Val')
    axes[1, 0].set_title('Standard ANN - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot([a*100 for a in std_trainer.bp_history['train_acc']], label='Train')
    if std_trainer.bp_history['val_acc']:
        axes[1, 1].plot([a*100 for a in std_trainer.bp_history['val_acc']], label='Val')
    axes[1, 1].set_title('Standard ANN - Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('%')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
