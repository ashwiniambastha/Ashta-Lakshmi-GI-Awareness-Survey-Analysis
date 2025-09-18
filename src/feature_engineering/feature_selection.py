"""
Feature selection module for Ashta Lakshmi GI Survey
Implements various feature selection techniques to identify most important features
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    f_classif, f_regression, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    Class for comprehensive feature selection using multiple techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureSelector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.selected_features = {}
        self.feature_importance_scores = {}
        
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature selection pipeline
        
        Args:
            df: Input dataframe with engineered features
            
        Returns:
            DataFrame with selected features
        """
        self.logger.info("Starting feature selection...")
        
        # Prepare data for feature selection
        X, y, feature_names = self._prepare_data_for_selection(df)
        
        # Remove low variance features
        X_variance, selected_variance = self._remove_low_variance_features(X, feature_names)
        
        # Statistical feature selection
        selected_statistical = self._statistical_feature_selection(X_variance, y, selected_variance)
        
        # Model-based feature selection
        selected_model_based = self._model_based_feature_selection(X_variance, y, selected_variance)
        
        # Recursive feature elimination
        selected_rfe = self._recursive_feature_elimination(X_variance, y, selected_variance)
        
        # Combine selection methods
        final_features = self._combine_selection_methods(
            selected_variance, selected_statistical, selected_model_based, selected_rfe
        )
        
        # Create final dataset with selected features
        selected_df = self._create_selected_dataset(df, final_features)
        
        # Generate feature selection report
        self._generate_selection_report(final_features, df.shape[1])
        
        self.logger.info(f"Feature selection completed. Selected {len(final_features)} features from {len(feature_names)}")
        return selected_df
    
    def _prepare_data_for_selection(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for feature selection
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Define target variable (assuming GI_Aware_Binary is the main target)
        target_col = 'GI_Aware_Binary'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in dataframe")
        
        # Select features (exclude target and identifier columns)
        exclude_cols = [
            target_col, 'Artisan_ID', 'State', 'Gender', 'GI_Aware', 
            'Received_Subsidy', 'Uses_Ecommerce', 'Age_Group', 'Experience_Group'
        ]
        
        # Only select numeric columns for feature selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Filter out any remaining non-numeric or problematic columns
        final_feature_cols = []
        for col in feature_cols:
            try:
                # Test if column can be converted to float
                pd.to_numeric(df[col], errors='raise')
                final_feature_cols.append(col)
            except:
                self.logger.warning(f"Skipping non-numeric column: {col}")
        
        if len(final_feature_cols) == 0:
            raise ValueError("No suitable numeric features found for selection")
        
        # Create feature matrix
        X = df[final_feature_cols].values.astype(float)
        y = df[target_col].values.astype(int)
        
        # Handle missing values if any
        if pd.isna(X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
            self.logger.info("Applied median imputation to handle missing values")
        
        self.logger.info(f"Prepared {len(final_feature_cols)} numeric features for selection")
        return X, y, final_feature_cols
    
    def _remove_low_variance_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Remove features with low variance
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Tuple of (filtered X, selected feature names)
        """
        # Remove features with zero or very low variance
        variance_threshold = VarianceThreshold(threshold=0.01)
        X_variance = variance_threshold.fit_transform(X)
        
        # Get selected feature names
        selected_features = [feature_names[i] for i in range(len(feature_names)) 
                           if variance_threshold.get_support()[i]]
        
        removed_count = len(feature_names) - len(selected_features)
        self.logger.info(f"Removed {removed_count} low-variance features")
        
        self.selected_features['variance_threshold'] = selected_features
        return X_variance, selected_features
    
    def _statistical_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> List[str]:
        """
        Select features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            List of selected feature names
        """
        # F-score based selection
        k_best = self.config.get('features', {}).get('feature_selection_k', 10)
        
        # Select k best features
        selector_k_best = SelectKBest(score_func=f_classif, k=min(k_best, len(feature_names)))
        X_k_best = selector_k_best.fit_transform(X, y)
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) 
                           if selector_k_best.get_support()[i]]
        
        # Store feature scores
        feature_scores = selector_k_best.scores_
        self.feature_importance_scores['f_score'] = dict(zip(feature_names, feature_scores))
        
        # Select top percentile features
        selector_percentile = SelectPercentile(score_func=f_classif, percentile=75)
        selector_percentile.fit(X, y)
        
        percentile_features = [feature_names[i] for i in range(len(feature_names)) 
                             if selector_percentile.get_support()[i]]
        
        # Combine both selections
        combined_statistical = list(set(selected_features + percentile_features))
        
        self.selected_features['statistical'] = combined_statistical
        self.logger.info(f"Statistical selection: {len(combined_statistical)} features")
        
        return combined_statistical
    
    def _model_based_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> List[str]:
        """
        Select features using model-based importance
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            List of selected feature names
        """
        selected_features = []
        
        # Random Forest feature importance
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector.fit(X, y)
        
        rf_importance = rf_selector.feature_importances_
        self.feature_importance_scores['random_forest'] = dict(zip(feature_names, rf_importance))
        
        # Select features above median importance
        median_importance = np.median(rf_importance)
        rf_selected = [feature_names[i] for i in range(len(feature_names)) 
                      if rf_importance[i] > median_importance]
        
        selected_features.extend(rf_selected)
        
        # Extra Trees feature importance
        et_selector = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et_selector.fit(X, y)
        
        et_importance = et_selector.feature_importances_
        self.feature_importance_scores['extra_trees'] = dict(zip(feature_names, et_importance))
        
        # Select top features from Extra Trees
        et_threshold = np.percentile(et_importance, 75)
        et_selected = [feature_names[i] for i in range(len(feature_names)) 
                      if et_importance[i] > et_threshold]
        
        selected_features.extend(et_selected)
        
        # LASSO feature selection
        try:
            lasso_selector = SelectFromModel(LassoCV(cv=5, random_state=42))
            lasso_selector.fit(X, y)
            
            lasso_selected = [feature_names[i] for i in range(len(feature_names)) 
                            if lasso_selector.get_support()[i]]
            selected_features.extend(lasso_selected)
            
            # Store LASSO coefficients
            lasso_coefs = lasso_selector.estimator_.coef_
            self.feature_importance_scores['lasso'] = dict(zip(feature_names, np.abs(lasso_coefs)))
            
        except Exception as e:
            self.logger.warning(f"LASSO selection failed: {str(e)}")
        
        # Remove duplicates
        model_based_features = list(set(selected_features))
        
        self.selected_features['model_based'] = model_based_features
        self.logger.info(f"Model-based selection: {len(model_based_features)} features")
        
        return model_based_features
    
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str]) -> List[str]:
        """
        Select features using recursive feature elimination
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            List of selected feature names
        """
        # RFE with cross-validation
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Determine optimal number of features
        selector_rfecv = RFECV(estimator=estimator, step=1, cv=3, scoring='accuracy', 
                              min_features_to_select=5)
        
        try:
            selector_rfecv.fit(X, y)
            
            rfe_selected = [feature_names[i] for i in range(len(feature_names)) 
                          if selector_rfecv.get_support()[i]]
            
            # Store RFE rankings
            rfe_rankings = selector_rfecv.ranking_
            self.feature_importance_scores['rfe_ranking'] = dict(zip(feature_names, rfe_rankings))
            
            self.selected_features['rfe'] = rfe_selected
            self.logger.info(f"RFE selection: {len(rfe_selected)} features (optimal: {selector_rfecv.n_features_})")
            
            return rfe_selected
            
        except Exception as e:
            self.logger.warning(f"RFE selection failed: {str(e)}")
            # Fallback to regular RFE
            n_features_to_select = min(15, len(feature_names) // 2)
            selector_rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            selector_rfe.fit(X, y)
            
            rfe_selected = [feature_names[i] for i in range(len(feature_names)) 
                          if selector_rfe.get_support()[i]]
            
            return rfe_selected
    
    def _combine_selection_methods(self, variance_features: List[str], 
                                 statistical_features: List[str],
                                 model_based_features: List[str], 
                                 rfe_features: List[str]) -> List[str]:
        """
        Combine different feature selection methods
        
        Args:
            variance_features: Features selected by variance threshold
            statistical_features: Features selected by statistical tests
            model_based_features: Features selected by model importance
            rfe_features: Features selected by RFE
            
        Returns:
            List of final selected features
        """
        # Count votes for each feature
        all_features = set(variance_features)
        feature_votes = {}
        
        for feature in all_features:
            votes = 0
            if feature in statistical_features:
                votes += 1
            if feature in model_based_features:
                votes += 1
            if feature in rfe_features:
                votes += 1
            feature_votes[feature] = votes
        
        # Select features with at least 2 votes or top features by importance
        min_votes = 2
        selected_by_votes = [feature for feature, votes in feature_votes.items() if votes >= min_votes]
        
        # If too few features selected, include top features from random forest
        if len(selected_by_votes) < 10:
            rf_importance = self.feature_importance_scores.get('random_forest', {})
            top_rf_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            top_rf_names = [name for name, _ in top_rf_features]
            selected_by_votes.extend([f for f in top_rf_names if f not in selected_by_votes])
        
        # Remove duplicates and ensure reasonable number of features
        final_features = list(set(selected_by_votes))[:25]  # Cap at 25 features
        
        self.selected_features['final'] = final_features
        return final_features
    
    def _create_selected_dataset(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        Create dataset with only selected features
        
        Args:
            df: Original dataframe
            selected_features: List of selected feature names
            
        Returns:
            DataFrame with selected features plus target and identifiers
        """
        # Include essential columns
        essential_cols = ['Artisan_ID', 'State', 'Gender', 'GI_Aware_Binary', 
                         'Received_Subsidy_Binary', 'Uses_Ecommerce_Binary']
        
        # Combine selected features with essential columns
        final_cols = essential_cols + [col for col in selected_features if col in df.columns]
        final_cols = list(set(final_cols))  # Remove duplicates
        
        selected_df = df[final_cols].copy()
        
        return selected_df
    
    def _generate_selection_report(self, selected_features: List[str], original_feature_count: int) -> None:
        """
        Generate feature selection report
        
        Args:
            selected_features: List of selected features
            original_feature_count: Number of original features
        """
        report_path = Path(self.config['data']['output_dir']) / 'reports' / 'feature_selection_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("FEATURE SELECTION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Original number of features: {original_feature_count}\n")
            f.write(f"Final number of selected features: {len(selected_features)}\n")
            f.write(f"Feature reduction: {((original_feature_count - len(selected_features)) / original_feature_count * 100):.1f}%\n\n")
            
            f.write("SELECTION METHODS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for method, features in self.selected_features.items():
                f.write(f"{method}: {len(features)} features\n")
            
            f.write(f"\nFINAL SELECTED FEATURES:\n")
            f.write("-" * 30 + "\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"{i}. {feature}\n")
            
            if 'random_forest' in self.feature_importance_scores:
                f.write(f"\nTOP 10 FEATURES BY RANDOM FOREST IMPORTANCE:\n")
                f.write("-" * 45 + "\n")
                rf_scores = self.feature_importance_scores['random_forest']
                top_features = sorted(rf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, score) in enumerate(top_features, 1):
                    f.write(f"{i}. {feature}: {score:.4f}\n")
        
        self.logger.info(f"Feature selection report saved to: {report_path}")
    
    def get_feature_importance(self, method: str = 'random_forest') -> Dict[str, float]:
        """
        Get feature importance scores from specified method
        
        Args:
            method: Method name for importance scores
            
        Returns:
            Dictionary of feature importance scores
        """
        return self.feature_importance_scores.get(method, {})
