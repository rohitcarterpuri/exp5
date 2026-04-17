import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
from typing import Tuple, Dict, Any

class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load churn dataset"""
        # Example dataset structure - adjust based on your data
        df = pd.read_csv(filepath)
        return df
    
    def preprocess(self, df: pd.DataFrame, target_col: str = 'churn') -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the data"""
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self.feature_names = X.columns.tolist()
        
        return X.values, y.values
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance"""
        strategy = self.config['data']['sampling_strategy']
        
        if strategy == 'smote':
            sampler = SMOTE(random_state=self.config['data']['random_state'])
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        elif strategy == 'random_oversample':
            sampler = RandomOverSampler(random_state=self.config['data']['random_state'])
            X_resampled, y_resampled = sampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
            
        return X_resampled, y_resampled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train, validation, test sets"""
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['validation_size']
        random_state = self.config['data']['random_state']
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Split train+val into train and validation
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative_size, 
            random_state=random_state, stratify=y_train_val
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def save_preprocessors(self, path: str):
        """Save scaler and label encoders"""
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(self.feature_names, f"{path}/feature_names.pkl")
