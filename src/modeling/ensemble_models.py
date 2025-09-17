"""
Ensemble modeling module for Ashta Lakshmi GI Survey
Implements advanced machine learning models with hyperparameter optimization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class EnsembleModeler:
    """
    Class for training and optimizing ensemble machine learning models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EnsembleModeler with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_results = {}
        self.best_models = {}
        
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter optimization
        
        Args:
            df: Input dataframe with selected features
            
        Returns:
            Dictionary containing trained models and results
        """
        self.logger.info("Starting model training...")
        
        # Prepare data for modeling
        X_train, X_test, y_train, y_test = self._prepare_modeling_data(df)
        
        # Train
