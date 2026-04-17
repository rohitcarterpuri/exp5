import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from typing import Dict, Any, Tuple, List

class ANNClassifier:
    def __init__(self, input_shape: int, config: Dict[str, Any]):
        self.input_shape = input_shape
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self, hidden_layers: List[int] = None, 
                    dropout_rate: float = None,
                    learning_rate: float = None) -> keras.Model:
        """Build ANN model architecture"""
        if hidden_layers is None:
            hidden_layers = self.config['model']['architecture']['hidden_layers']
        if dropout_rate is None:
            dropout_rate = self.config['model']['architecture']['dropout_rate']
        if learning_rate is None:
            learning_rate = self.config['training']['learning_rate']
            
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_shape,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation=self.config['model']['architecture']['activation']))
            
            if self.config['model']['architecture']['batch_normalization']:
                model.add(layers.BatchNormalization())
                
            model.add(layers.Dropout(dropout_rate))
            
        # Output layer
        model.add(layers.Dense(1, activation=self.config['model']['architecture']['output_activation']))
        
        # Compile model
        optimizer = self._get_optimizer(learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.config['training']['loss'],
            metrics=['accuracy', tf.keras.metrics.Precision(), 
                    tf.keras.metrics.Recall(), tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def _get_optimizer(self, learning_rate: float):
        """Get optimizer based on config"""
        optimizer_name = self.config['training']['optimizer'].lower()
        
        if optimizer_name == 'adam':
            return keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            return keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            return keras.optimizers.Adam(learning_rate=learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              callbacks_list: List[callbacks.Callback] = None) -> keras.callbacks.History:
        """Train the model"""
        batch_size = self.config['training']['batch_size']
        epochs = self.config['training']['epochs']
        
        if callbacks_list is None:
            callbacks_list = []
            
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_classes(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Make class predictions"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)
    
    def save_model(self, filepath: str):
        """Save model"""
        self.model.save(filepath)
        
    def load_model(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
