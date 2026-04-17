import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras import callbacks
from typing import Dict, Any, List, Tuple
import json
import os

class CrossValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        
    def cross_validate(self, model_class, X: np.ndarray, y: np.ndarray, 
                       callbacks_list: List[callbacks.Callback]) -> Dict[str, Any]:
        """Perform k-fold cross validation"""
        n_folds = self.config['cross_validation']['n_folds']
        scoring = self.config['cross_validation']['scoring']
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                              random_state=self.config['data']['random_state'])
        
        fold_results = {metric: [] for metric in scoring}
        fold_histories = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nTraining Fold {fold + 1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and train model
            model = model_class(X.shape[1], self.config)
            model.build_model()
            
            history = model.train(X_train, y_train, X_val, y_val, callbacks_list)
            fold_histories.append(history)
            
            # Evaluate
            y_pred_prob = model.predict(X_val)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            if 'accuracy' in scoring:
                fold_results['accuracy'].append(accuracy_score(y_val, y_pred))
            if 'precision' in scoring:
                fold_results['precision'].append(precision_score(y_val, y_pred))
            if 'recall' in scoring:
                fold_results['recall'].append(recall_score(y_val, y_pred))
            if 'f1' in scoring:
                fold_results['f1'].append(f1_score(y_val, y_pred))
            if 'roc_auc' in scoring:
                fold_results['roc_auc'].append(roc_auc_score(y_val, y_pred_prob))
        
        # Aggregate results
        aggregated_results = {}
        for metric, values in fold_results.items():
            aggregated_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        
        self.results = aggregated_results
        return aggregated_results

class GridSearchOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_params = {}
        self.best_score = -np.inf
        
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    param_grid: Dict[str, List]) -> Dict[str, Any]:
        """Perform grid search for hyperparameters"""
        results = []
        
        from itertools import product
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        
        for combination in product(*values):
            params = dict(zip(keys, combination))
            
            print(f"\nTesting parameters: {params}")
            
            # Build model with current parameters
            from src.models.ann_model import ANNClassifier
            model = ANNClassifier(X_train.shape[1], self.config)
            
            # Handle nested parameters
            hidden_layers = params.get('hidden_layers', self.config['model']['architecture']['hidden_layers'])
            dropout_rate = params.get('dropout_rate', self.config['model']['architecture']['dropout_rate'])
            learning_rate = params.get('learning_rate', self.config['training']['learning_rate'])
            batch_size = params.get('batch_size', self.config['training']['batch_size'])
            
            # Temporarily override config
            original_batch_size = self.config['training']['batch_size']
            self.config['training']['batch_size'] = batch_size
            
            model.build_model(hidden_layers=hidden_layers, 
                            dropout_rate=dropout_rate,
                            learning_rate=learning_rate)
            
            # Train model
            history = model.train(X_train, y_train, X_val, y_val, callbacks_list=[])
            
            # Evaluate
            y_pred_prob = model.predict(X_val)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(y_val, y_pred_prob)
            
            results.append({
                'params': params,
                'val_auc': val_auc,
                'history': history.history
            })
            
            # Update best
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.best_params = params
                
            # Restore original config
            self.config['training']['batch_size'] = original_batch_size
            
            print(f"Validation AUC: {val_auc:.4f}")
        
        # Sort results by validation AUC
        results.sort(key=lambda x: x['val_auc'], reverse=True)
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best validation AUC: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': results
        }
