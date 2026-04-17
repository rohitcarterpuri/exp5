import tensorflow as tf
from tensorflow.keras import callbacks
import os
from datetime import datetime
from typing import Dict, Any

class CheckpointManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpoint_dir = config['checkpoint']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def create_model_checkpoint(self) -> callbacks.ModelCheckpoint:
        """Create model checkpoint callback"""
        monitor = self.config['checkpoint']['monitor']
        mode = self.config['checkpoint']['mode']
        save_best_only = self.config['checkpoint']['save_best_only']
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"model_epoch_{{epoch:02d}}_{{{monitor}:.4f}}.h5"
        )
        
        return callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode=mode,
            verbose=1
        )
    
    def create_early_stopping(self, patience: int = 10) -> callbacks.EarlyStopping:
        """Create early stopping callback"""
        monitor = self.config['checkpoint']['monitor']
        mode = self.config['checkpoint']['mode']
        
        return callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            restore_best_weights=True,
            verbose=1
        )
    
    def create_reduce_lr(self, patience: int = 5, factor: float = 0.5) -> callbacks.ReduceLROnPlateau:
        """Create learning rate reduction callback"""
        return callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            min_lr=1e-7,
            verbose=1
        )
    
    def create_tensorboard(self) -> callbacks.TensorBoard:
        """Create TensorBoard callback"""
        log_dir = os.path.join(
            self.config['logging']['log_dir'],
            datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(log_dir, exist_ok=True)
        
        return callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
    
    def get_all_callbacks(self) -> list:
        """Get all callbacks"""
        callbacks_list = [
            self.create_model_checkpoint(),
            self.create_early_stopping(),
            self.create_reduce_lr(),
            self.create_tensorboard()
        ]
        
        return callbacks_list
