import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from typing import Dict, Any

class MetricsCalculator:
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_prob: np.ndarray) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_prob)
        }
        
        return metrics
    
    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report"""
        return classification_report(y_true, y_pred, target_names=['No Churn', 'Churn'])
