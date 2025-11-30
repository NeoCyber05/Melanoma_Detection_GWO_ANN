"""
Grey Wolf Optimizer (GWO) - Thuật toán tối ưu hóa bầy sói xám
"""

import numpy as np
from tqdm import tqdm
import torch


class GreyWolfOptimizer:
    
    def __init__(self, objective_func, dim, population_size=30, max_iter=100,
                 lower_bound=-1.0, upper_bound=1.0, verbose=True):
        self.objective_func = objective_func
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Bounds
        if np.isscalar(lower_bound):
            self.lower_bound = lower_bound * np.ones(dim)
        else:
            self.lower_bound = np.array(lower_bound)
            
        if np.isscalar(upper_bound):
            self.upper_bound = upper_bound * np.ones(dim)
        else:
            self.upper_bound = np.array(upper_bound)
        
        # Alpha, Beta, Delta (top 3 solutions)
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')
        
        self.convergence_curve = []
    
    def _init_population(self):
        return np.random.uniform(
            self.lower_bound, self.upper_bound,
            size=(self.population_size, self.dim)
        )
    
    def _clip(self, pos):
        return np.clip(pos, self.lower_bound, self.upper_bound)
    
    def optimize(self):
        population = self._init_population()
        
        iterator = tqdm(range(self.max_iter), desc="GWO") if self.verbose else range(self.max_iter)
        
        for t in iterator:
            # Đánh giá fitness
            for i in range(self.population_size):
                population[i] = self._clip(population[i])
                fitness = self.objective_func(population[i])
                
                # Cập nhật alpha, beta, delta
                if fitness < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy() if self.beta_pos is not None else None
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy() if self.alpha_pos is not None else None
                    self.alpha_score, self.alpha_pos = fitness, population[i].copy()
                elif fitness < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy() if self.beta_pos is not None else None
                    self.beta_score, self.beta_pos = fitness, population[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score, self.delta_pos = fitness, population[i].copy()
            
            # Hệ số a giảm từ 2 về 0
            a = 2 - t * (2 / self.max_iter)
            
            # Cập nhật vị trí các sói
            for i in range(self.population_size):
                # Theo alpha
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = np.abs(C1 * self.alpha_pos - population[i])
                X1 = self.alpha_pos - A1 * D_alpha
                
                # Theo beta
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = np.abs(C2 * self.beta_pos - population[i])
                X2 = self.beta_pos - A2 * D_beta
                
                # Theo delta
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = np.abs(C3 * self.delta_pos - population[i])
                X3 = self.delta_pos - A3 * D_delta
                
                # Vị trí mới = trung bình
                population[i] = (X1 + X2 + X3) / 3
            
            self.convergence_curve.append(self.alpha_score)
            
            if self.verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'MSE': f'{self.alpha_score:.6f}'})
        
        return self.alpha_pos, self.alpha_score
    
    def get_convergence_curve(self):
        return np.array(self.convergence_curve)


def create_objective_function(model, X, y, device='cpu'):
    """Tạo hàm objective (MSE) cho GWO"""
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    def objective(weights):
        model.set_weights_from_vector(weights)
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # One-hot
            y_onehot = torch.zeros(len(y_tensor), 2).to(device)
            y_onehot.scatter_(1, y_tensor.unsqueeze(1), 1)
            
            # MSE = 1/2 * mean((pred - true)^2)
            mse = 0.5 * torch.mean((probs - y_onehot) ** 2).item()
        
        return mse
    
    return objective


if __name__ == "__main__":
    # Test nhanh với sphere function
    def sphere(x):
        return np.sum(x ** 2)
    
    gwo = GreyWolfOptimizer(sphere, dim=10, population_size=20, max_iter=30)
    best_pos, best_score = gwo.optimize()
    print(f"Best score: {best_score:.6f} (expected: 0)")
