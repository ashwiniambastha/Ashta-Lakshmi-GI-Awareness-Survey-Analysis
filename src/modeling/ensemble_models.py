"""
Ensemble modeling module for Ashta Lakshmi GI Survey
Implements advanced machine learning models with hyperparameter optimization
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, StackingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import scikit-optimize, fallback to GridSearch if not available
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPTIMIZATION_AVAILABLE = True
except ImportError:
    BAYESIAN_OPTIMIZATION_AVAILABLE = False
    logging.getLogger(__name__).warning("scikit-optimize not available. Using GridSearchCV instead of Bayesian optimization.")

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
        
        # Train base models
        base_models = self._train_base_models(X_train, X_test, y_train, y_test)
        
        # Train ensemble models
        ensemble_models = self._train_ensemble_models(X_train, X_test, y_train, y_test, base_models)
        
        # Hyperparameter optimization
        optimized_models = self._optimize_hyperparameters(X_train, y_train)
        
        # Combine all results
        all_models = {**base_models, **ensemble_models, **optimized_models}
        
        # Select best models
        self.best_models = self._select_best_models(all_models)
        
        # Save models
        self._save_models()
        
        self.logger.info("Model training completed")
        return {
            'models': all_models,
            'best_models': self.best_models,
            'test_data': (X_test, y_test)
        }
    
    def _prepare_modeling_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for modeling
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of train/test splits
        """
        # Define target variable
        target_col = 'GI_Aware_Binary'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")
        
        # Select feature columns (exclude target and identifiers)
        exclude_cols = [target_col, 'Artisan_ID', 'State', 'Gender']
        
        # Only select numeric columns to avoid dtype issues
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Filter out any problematic columns
        final_feature_cols = []
        for col in feature_cols:
            try:
                # Ensure column is numeric and finite
                col_data = pd.to_numeric(df[col], errors='coerce')
                if not col_data.isna().all():  # Don't include columns that are all NaN
                    final_feature_cols.append(col)
            except:
                self.logger.warning(f"Skipping problematic column: {col}")
        
        if len(final_feature_cols) == 0:
            raise ValueError("No suitable numeric features found for modeling")
        
        X = df[final_feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)
        
        # Handle any remaining missing values
        if pd.isna(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            self.logger.info("Applied imputation to handle missing values in modeling data")
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        self.logger.info(f"Using {len(final_feature_cols)} features for modeling")
        return X_train, X_test, y_train, y_test
    
    def _train_base_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Train base machine learning models
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data splits
            
        Returns:
            Dictionary containing trained base models and their results
        """
        base_models = {}
        
        # Random Forest
        rf_params = self.config.get('models', {}).get('random_forest', {})
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        rf_results = self._evaluate_model(rf, X_test, y_test, 'Random Forest')
        base_models['random_forest'] = {'model': rf, 'results': rf_results}
        
        # Gradient Boosting
        gb_params = self.config.get('models', {}).get('gradient_boosting', {})
        gb = GradientBoostingClassifier(**gb_params)
        gb.fit(X_train, y_train)
        gb_results = self._evaluate_model(gb, X_test, y_test, 'Gradient Boosting')
        base_models['gradient_boosting'] = {'model': gb, 'results': gb_results}
        
        # SVM
        svm_params = self.config.get('models', {}).get('svm', {})
        svm = SVC(**svm_params)
        svm.fit(X_train, y_train)
        svm_results = self._evaluate_model(svm, X_test, y_test, 'SVM')
        base_models['svm'] = {'model': svm, 'results': svm_results}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        lr_results = self._evaluate_model(lr, X_test, y_test, 'Logistic Regression')
        base_models['logistic_regression'] = {'model': lr, 'results': lr_results}
        
        # Extra Trees
        et = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et.fit(X_train, y_train)
        et_results = self._evaluate_model(et, X_test, y_test, 'Extra Trees')
        base_models['extra_trees'] = {'model': et, 'results': et_results}
        
        self.logger.info("Base models trained")
        return base_models
    
    def _train_ensemble_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                              y_train: np.ndarray, y_test: np.ndarray,
                              base_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ensemble models using base models
        
        Args:
            X_train, X_test, y_train, y_test: Train/test data splits
            base_models: Dictionary of trained base models
            
        Returns:
            Dictionary containing trained ensemble models
        """
        ensemble_models = {}
        
        # Voting Classifier
        voting_estimators = [
            ('rf', base_models['random_forest']['model']),
            ('gb', base_models['gradient_boosting']['model']),
            ('lr', base_models['logistic_regression']['model'])
        ]
        
        voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
        voting_clf.fit(X_train, y_train)
        voting_results = self._evaluate_model(voting_clf, X_test, y_test, 'Voting Classifier')
        ensemble_models['voting_classifier'] = {'model': voting_clf, 'results': voting_results}
        
        # Stacking Classifier
        stacking_estimators = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=50, random_state=42))
        ]
        
        stacking_clf = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
        stacking_clf.fit(X_train, y_train)
        stacking_results = self._evaluate_model(stacking_clf, X_test, y_test, 'Stacking Classifier')
        ensemble_models['stacking_classifier'] = {'model': stacking_clf, 'results': stacking_results}
        
        self.logger.info("Ensemble models trained")
        return ensemble_models
    
    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization or Grid search
        
        Args:
            X_train, y_train: Training data
            
        Returns:
            Dictionary containing optimized models
        """
        optimized_models = {}
        
        # Random Forest hyperparameter optimization
        if BAYESIAN_OPTIMIZATION_AVAILABLE:
            rf_search_spaces = {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', None])
            }
            
            try:
                rf_search = BayesSearchCV(
                    RandomForestClassifier(random_state=42),
                    rf_search_spaces,
                    n_iter=30,
                    cv=3,
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_search.fit(X_train, y_train)
                optimized_models['optimized_random_forest'] = {
                    'model': rf_search.best_estimator_,
                    'best_params': rf_search.best_params_,
                    'best_score': rf_search.best_score_
                }
                
                self.logger.info(f"Bayesian RF optimization completed. Best score: {rf_search.best_score_:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Bayesian RF optimization failed: {str(e)}")
        else:
            # Fallback to Grid Search
            rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
            
            try:
                rf_grid = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    rf_param_grid,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                rf_grid.fit(X_train, y_train)
                optimized_models['optimized_random_forest'] = {
                    'model': rf_grid.best_estimator_,
                    'best_params': rf_grid.best_params_,
                    'best_score': rf_grid.best_score_
                }
                
                self.logger.info(f"Grid RF optimization completed. Best score: {rf_grid.best_score_:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Grid RF optimization failed: {str(e)}")
        
        return optimized_models
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (if model supports predict_proba)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = None
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        self.logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def _select_best_models(self, all_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select best performing models
        
        Args:
            all_models: Dictionary of all trained models
            
        Returns:
            Dictionary containing best models
        """
        best_models = {}
        
        # Find best model by accuracy
        best_accuracy = 0
        best_accuracy_model = None
        
        # Find best model by F1 score
        best_f1 = 0
        best_f1_model = None
        
        for model_name, model_info in all_models.items():
            if 'results' in model_info:
                accuracy = model_info['results']['accuracy']
                f1_score = model_info['results']['f1_score']
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_model = (model_name, model_info)
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_f1_model = (model_name, model_info)
        
        if best_accuracy_model:
            best_models['best_accuracy'] = best_accuracy_model
        
        if best_f1_model:
            best_models['best_f1'] = best_f1_model
        
        return best_models
    
    def _save_models(self) -> None:
        """
        Save trained models to disk
        """
        models_dir = Path(self.config['data']['output_dir']) / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best models
        for model_type, (model_name, model_info) in self.best_models.items():
            if 'model' in model_info:
                model_path = models_dir / f'{model_name}_{model_type}.joblib'
                joblib.dump(model_info['model'], model_path)
                self.logger.info(f"Saved {model_name} to {model_path}")
    
    def predict_new_data(self, X_new: np.ndarray, model_type: str = 'best_accuracy') -> np.ndarray:
        """
        Make predictions on new data using best model
        
        Args:
            X_new: New data for prediction
            model_type: Type of best model to use
            
        Returns:
            Predictions array
        """
        if model_type not in self.best_models:
            raise ValueError(f"Model type {model_type} not found in best models")
        
        model_name, model_info = self.best_models[model_type]
        model = model_info['model']
        
        predictions = model.predict(X_new)
        return predictions
    
    def get_feature_importance(self, model_type: str = 'best_accuracy') -> Dict[str, float]:
        """
        Get feature importance from best model (if available)
        
        Args:
            model_type: Type of best model
            
        Returns:
            Dictionary of feature importance scores
        """
        if model_type not in self.best_models:
            return {}
        
        model_name, model_info = self.best_models[model_type]
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return {}
