"""
Hybrid GWO-ANN Trainer
Phase 1: GWO tìm trọng số tối ưu
Phase 2: Back-propagation tinh chỉnh
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


class HybridGWOANNTrainer:
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.gwo_history = []
        self.bp_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def gwo_phase(self, X_train, y_train, population_size=30, max_iter=50, verbose=True):
        """Phase 1: Tối ưu trọng số bằng GWO"""
        print("\n" + "="*50)
        print("Phase 1: Grey Wolf Optimization")
        print("="*50)
        
        objective = create_objective_function(self.model, X_train, y_train, self.device)
        dim = self.model.get_num_params()
        lb, ub = self.model.get_weight_bounds()
        
        print(f"Parameters: {dim}, Population: {population_size}, Iterations: {max_iter}")
        
        gwo = GreyWolfOptimizer(objective, dim, population_size, max_iter, lb, ub, verbose)
        
        start = time.time()
        best_weights, best_mse = gwo.optimize()
        
        self.gwo_history = gwo.get_convergence_curve().tolist()
        self.model.set_weights_from_vector(best_weights)
        
        acc = calculate_accuracy(self.model, X_train, y_train, self.device)
        print(f"\nGWO done in {time.time()-start:.2f}s | MSE: {best_mse:.6f} | Acc: {acc*100:.2f}%")
        
        return best_weights, best_mse
    
    def bp_phase(self, X_train, y_train, X_val=None, y_val=None,
                 epochs=100, batch_size=32, lr=0.001, patience=10, verbose=True):
        """Phase 2: Fine-tune bằng Back-propagation"""
        print("\n" + "="*50)
        print("Phase 2: Back-propagation")
        print("="*50)
        
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
            self.bp_history['train_loss'].append(train_loss)
            self.bp_history['train_acc'].append(train_acc)
            
            # Validation
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
                    print(f"Early stop at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)
                    break
                
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val: {val_loss:.4f} {val_acc*100:.1f}%")
            else:
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}%")
        
        print(f"BP done in {time.time()-start:.2f}s")
    
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
              gwo_pop=30, gwo_iter=50, bp_epochs=100, bp_batch=32, bp_lr=0.001, patience=10):
        """Full hybrid training"""
        print("\n" + "="*50)
        print("HYBRID GWO-ANN TRAINING")
        print("="*50)
        
        start = time.time()
        self.gwo_phase(X_train, y_train, gwo_pop, gwo_iter)
        self.bp_phase(X_train, y_train, X_val, y_val, bp_epochs, bp_batch, bp_lr, patience)
        print(f"\nTotal time: {time.time()-start:.2f}s")
    
    def evaluate(self, X_test, y_test):
        """Đánh giá trên test set"""
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
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
        
        print(f"Accuracy:    {acc*100:.2f}%")
        print(f"Precision:   {prec*100:.2f}%")
        print(f"Recall:      {rec*100:.2f}%")
        print(f"F1-Score:    {f1*100:.2f}%")
        print(f"Sensitivity: {sens*100:.2f}%")
        print(f"Specificity: {spec*100:.2f}%")
        print(f"\nConfusion Matrix:\n{cm}")
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'sensitivity': sens, 'specificity': spec, 'cm': cm}
    
    def plot_history(self, save_path=None):
        """Vẽ training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # GWO convergence
        if self.gwo_history:
            axes[0].plot(self.gwo_history, 'b-')
            axes[0].set_title('GWO Convergence')
            axes[0].set_xlabel('Iteration')
            axes[0].set_ylabel('MSE')
            axes[0].grid(True, alpha=0.3)
        
        # BP Loss
        if self.bp_history['train_loss']:
            axes[1].plot(self.bp_history['train_loss'], label='Train')
            if self.bp_history['val_loss']:
                axes[1].plot(self.bp_history['val_loss'], label='Val')
            axes[1].set_title('BP Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # BP Accuracy
        if self.bp_history['train_acc']:
            axes[2].plot([a*100 for a in self.bp_history['train_acc']], label='Train')
            if self.bp_history['val_acc']:
                axes[2].plot([a*100 for a in self.bp_history['val_acc']], label='Val')
            axes[2].set_title('BP Accuracy')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('%')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        plt.show()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


class StandardANNTrainer:
    """ANN thuần - chỉ dùng Back-propagation (để so sánh)"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, lr=0.001, patience=10, verbose=True):
        print("\n" + "="*50)
        print("STANDARD ANN TRAINING (BP only)")
        print("="*50)
        
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
                    print(f"Early stop at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)
                    break
                
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val: {val_loss:.4f} {val_acc*100:.1f}%")
            else:
                if verbose and (epoch+1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}%")
        
        print(f"Training done in {time.time()-start:.2f}s")
    
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
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
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
        
        print(f"Accuracy:    {acc*100:.2f}%")
        print(f"Precision:   {prec*100:.2f}%")
        print(f"Recall:      {rec*100:.2f}%")
        print(f"F1-Score:    {f1*100:.2f}%")
        print(f"Sensitivity: {sens*100:.2f}%")
        print(f"Specificity: {spec*100:.2f}%")
        print(f"\nConfusion Matrix:\n{cm}")
        
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
                'sensitivity': sens, 'specificity': spec, 'cm': cm}
    
    def plot_history(self, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        if self.history['train_loss']:
            axes[0].plot(self.history['train_loss'], label='Train')
            if self.history['val_loss']:
                axes[0].plot(self.history['val_loss'], label='Val')
            axes[0].set_title('Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        if self.history['train_acc']:
            axes[1].plot([a*100 for a in self.history['train_acc']], label='Train')
            if self.history['val_acc']:
                axes[1].plot([a*100 for a in self.history['val_acc']], label='Val')
            axes[1].set_title('Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('%')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(200, 50).astype(np.float32)
    y = np.random.randint(0, 2, 200)
    
    # Test Hybrid
    model1 = MelanomaANN(50, 32)
    trainer1 = HybridGWOANNTrainer(model1)
    trainer1.train(X, y, gwo_pop=10, gwo_iter=10, bp_epochs=20)
    
    # Test Standard
    model2 = MelanomaANN(50, 32)
    trainer2 = StandardANNTrainer(model2)
    trainer2.train(X, y, epochs=20)
