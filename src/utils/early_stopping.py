"""
Early stopping utility for training
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving
    """
    
    def __init__(
        self, 
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, validation_loss: float, model=None) -> bool:
        """
        Check if early stopping condition is met
        
        Args:
            validation_loss: Current validation loss
            model: Model to save best weights (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = validation_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                logger.info(f"Validation loss improved to {validation_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered!")
                
                # Restore best weights if requested
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Restored best weights")
        
        return self.early_stop
