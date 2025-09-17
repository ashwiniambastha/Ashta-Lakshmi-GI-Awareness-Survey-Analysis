"""
Correlation analysis module for Ashta Lakshmi GI Survey
Advanced correlation analysis and relationship discovery
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Class for comprehensive correlation and relationship analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CorrelationAnalyzer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing correlation analysis results
        """
        self.logger.info("Starting correlation analysis...")
        
        results = {}
        
        # Prepare data for correlation analysis
        corr_data = self._prepare_correlation_data(df)
        
        # Pearson correlation matrix
        results['pearson_matrix'] = self._calculate_pearson_correlation(corr_data)
        
        # Spearman correlation matrix
        results['spearman_matrix'] = self._calculate_spearman_correlation(corr_data)
        
        # Kendall tau correlation
        results['kendall_matrix'] = self._calculate_kendall_correlation(corr_data)
        
        # Mutual information analysis
        results['mutual_information'] = self._calculate_mutual_information(df)
        
        # Correlation significance testing
        results['correlation_significance'] = self._test_correlation_significance(corr_data)
        
        # Partial correlations
        results['partial_correlations'] = self._calculate_partial_correlations(corr_data)
        
        # Correlation patterns
        results['correlation_patterns'] = self._identify_correlation_patterns(results)
        
        self.logger.info("Correlation analysis completed")
        return results
    
    def _prepare_correlation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for correlation analysis
        
        Args:
            df: Input dataframe
            
        Returns:
            Prepared dataframe for correlation analysis
        """
        # Select numerical and binary variables
        numerical_cols = ['Age', 'Years_of_Experience']
        binary_cols = ['GI_Aware_Binary', 'Received_Subsidy_Binary', 'Uses_Ecommerce_Binary', 'Gender_Binary']
        
        # Create additional derived variables
        corr_df = df[numerical_cols + binary_cols].copy()
        
        # Add derived features for correlation analysis
        corr_df['Age_Experience_Ratio'] = corr_df['Age'] / (corr_df['Years_of_Experience'] + 1)
        corr_df['Experience_per_Age'] = corr_df['Years_of_Experience'] / corr_df['Age']
        
        # Add state encoding for correlation (numerical representation)
        state_mapping = {state: idx for idx, state in enumerate(df['State'].unique())}
        corr_df['State_Encoded'] = df['State'].map(state_mapping)
        
        return corr_df
    
    def _calculate_pearson_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Pearson correlation matrix with statistical significance
        
        Args:
            df: Correlation dataframe
            
        Returns:
            Dictionary containing Pearson correlation results
        """
        correlation_matrix = df.corr(method='pearson')
        
        # Calculate p-values for correlations
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    p_values.loc[col1, col2] = 0.0
                else:
                    corr_coef, p_val = pearsonr(df[col1], df[col2])
                    p_values.loc[col1, col2] = p_val
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'p_values': p_values.astype(float).to_dict(),
            'strong_correlations': self._extract_strong_correlations(correlation_matrix, p_values, threshold=0.5)
        }
    
    def _calculate_spearman_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Spearman rank correlation matrix
        
        Args:
            df: Correlation dataframe
            
        Returns:
            Dictionary containing Spearman correlation results
        """
        correlation_matrix = df.corr(method='spearman')
        
        # Calculate p-values for Spearman correlations
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    p_values.loc[col1, col2] = 0.0
                else:
                    corr_coef, p_val = spearmanr(df[col1], df[col2])
                    p_values.loc[col1, col2] = p_val
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'p_values': p_values.astype(float).to_dict(),
            'strong_correlations': self._extract_strong_correlations(correlation_matrix, p_values, threshold=0.5)
        }
    
    def _calculate_kendall_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Kendall tau correlation matrix
        
        Args:
            df: Correlation dataframe
            
        Returns:
            Dictionary containing Kendall correlation results
        """
        correlation_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 == col2:
                    correlation_matrix.loc[col1, col2] = 1.0
                    p_values.loc[col1, col2] = 0.0
                else:
                    tau, p_val = kendalltau(df[col1], df[col2])
                    correlation_matrix.loc[col1, col2] = tau
                    p_values.loc[col1, col2] = p_val
        
        return {
            'correlation_matrix': correlation_matrix.astype(float).to_dict(),
            'p_values': p_values.astype(float).to_dict(),
            'strong_correlations': self._extract_strong_correlations(correlation_matrix.astype(float), 
                                                                    p_values.astype(float), threshold=0.3)
        }
    
    def _calculate_mutual_information(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate mutual information between variables
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing mutual information results
        """
        mutual_info_results = {}
        
        # Prepare features and targets
        numerical_features = df[['Age', 'Years_of_Experience']]
        binary_targets = ['GI_Aware_Binary', 'Received_Subsidy_Binary', 'Uses_Ecommerce_Binary']
        
        # Calculate mutual information for classification targets
        for target in binary_targets:
            if target in df.columns:
                mi_scores = mutual_info_classif(numerical_features, df[target], random_state=42)
                mutual_info_results[f'{target}_mi'] = {
                    'Age': mi_scores[0],
                    'Years_of_Experience': mi_scores[1]
                }
        
        # Calculate mutual information for regression targets
        for target in ['Age', 'Years_of_Experience']:
            other_features = numerical_features.drop(columns=[target])
            if len(other_features.columns) > 0:
                mi_scores = mutual_info_regression(other_features, df[target], random_state=42)
                mutual_info_results[f'{target}_mi_regression'] = dict(zip(other_features.columns, mi_scores))
        
        return mutual_info_results
    
    def _test_correlation_significance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test statistical significance of correlations
        
        Args:
            df: Correlation dataframe
            
        Returns:
            Dictionary containing significance test results
        """
        significance_results = {}
        alpha = 0.05
        
        # Test significance for key variable pairs
        key_pairs = [
            ('Age', 'Years_of_Experience'),
            ('Age', 'GI_Aware_Binary'),
            ('Years_of_Experience', 'GI_Aware_Binary'),
            ('GI_Aware_Binary', 'Uses_Ecommerce_Binary'),
            ('Gender_Binary', 'GI_Aware_Binary')
        ]
        
        for var1, var2 in key_pairs:
            if var1 in df.columns and var2 in df.columns:
                # Pearson correlation test
                pearson_r, pearson_p = pearsonr(df[var1], df[var2])
                
                # Spearman correlation test
                spearman_r, spearman_p = spearmanr(df[var1], df[var2])
                
                significance_results[f'{var1}_vs_{var2}'] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'pearson_significant': pearson_p < alpha,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'spearman_significant': spearman_p < alpha,
                    'effect_size': self._interpret_correlation_strength(abs(pearson_r)),
                    'sample_size': len(df)
                }
        
        return significance_results
    
    def _calculate_partial_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate partial correlations controlling for confounding variables
        
        Args:
            df: Correlation dataframe
            
        Returns:
            Dictionary containing partial correlation results
        """
        from scipy.linalg import pinv
        
        partial_corr_results = {}
        
        # Calculate partial correlation between GI_Aware and other variables, controlling for Age
        if all(col in df.columns for col in ['GI_Aware_Binary', 'Years_of_Experience', 'Age']):
            partial_corr = self._partial_correlation(
                df['GI_Aware_Binary'], 
                df['Years_of_Experience'], 
                df['Age']
            )
            partial_corr_results['GI_Aware_vs_Experience_control_Age'] = partial_corr
        
        # Calculate partial correlation between Uses_Ecommerce and GI_Aware, controlling for Age
        if all(col in df.columns for col in ['Uses_Ecommerce_Binary', 'GI_Aware_Binary', 'Age']):
            partial_corr = self._partial_correlation(
                df['Uses_Ecommerce_Binary'], 
                df['GI_Aware_Binary'], 
                df['Age']
            )
            partial_corr_results['Ecommerce_vs_GI_Aware_control_Age'] = partial_corr
        
        return partial_corr_results
    
    def _partial_correlation(self, x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        """
        Calculate partial correlation between x and y, controlling for z
        
        Args:
            x: First variable
            y: Second variable
            z: Control variable
            
        Returns:
            Partial correlation coefficient
        """
        # Calculate correlations
        r_xy = pearsonr(x, y)[0]
        r_xz = pearsonr(x, z)[0]
        r_yz = pearsonr(y, z)[0]
        
        # Calculate partial correlation
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _extract_strong_correlations(self, corr_matrix: pd.DataFrame, p_values: pd.DataFrame, threshold: float) -> List[Dict[str, Any]]:
        """
        Extract strong correlations above threshold
        
        Args:
            corr_matrix: Correlation matrix
            p_values: P-values matrix
            threshold: Correlation threshold
            
        Returns:
            List of strong correlations
        """
        strong_correlations = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and diagonal
                    corr_value = corr_matrix.iloc[i, j]
                    p_value = p_values.iloc[i, j]
                    
                    if abs(corr_value) >= threshold and p_value < 0.05:
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr_value,
                            'p_value': p_value,
                            'strength': self._interpret_correlation_strength(abs(corr_value)),
                            'direction': 'positive' if corr_value > 0 else 'negative'
                        })
        
        return strong_correlations
    
    def _identify_correlation_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify patterns in correlation results
        
        Args:
            results: Correlation analysis results
            
        Returns:
            Dictionary containing correlation patterns
        """
        patterns = {}
        
        # Extract Pearson strong correlations
        pearson_strong = results.get('pearson_matrix', {}).get('strong_correlations', [])
        
        # Categorize correlations by strength
        patterns['by_strength'] = {
            'very_strong': [c for c in pearson_strong if abs(c['correlation']) >= 0.7],
            'strong': [c for c in pearson_strong if 0.5 <= abs(c['correlation']) < 0.7],
            'moderate': [c for c in pearson_strong if 0.3 <= abs(c['correlation']) < 0.5]
        }
        
        # Identify variables with most connections
        variable_connections = {}
        for corr in pearson_strong:
            for var in [corr['variable1'], corr['variable2']]:
                if var not in variable_connections:
                    variable_connections[var] = 0
                variable_connections[var] += 1
        
        patterns['most_connected_variables'] = sorted(
            variable_connections.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Identify correlation clusters
        patterns['correlation_clusters'] = self._identify_correlation_clusters(pearson_strong)
        
        return patterns
    
    def _identify_correlation_clusters(self, correlations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Identify clusters of highly correlated variables
        
        Args:
            correlations: List of correlation dictionaries
            
        Returns:
            Dictionary containing correlation clusters
        """
        clusters = {}
        processed_vars = set()
        
        for corr in correlations:
            var1, var2 = corr['variable1'], corr['variable2']
            
            if var1 not in processed_vars and var2 not in processed_vars:
                # Start a new cluster
                cluster_name = f"cluster_{len(clusters) + 1}"
                clusters[cluster_name] = [var1, var2]
                processed_vars.update([var1, var2])
            elif var1 in processed_vars or var2 in processed_vars:
                # Add to existing cluster
                for cluster_name, cluster_vars in clusters.items():
                    if var1 in cluster_vars or var2 in cluster_vars:
                        if var1 not in cluster_vars:
                            cluster_vars.append(var1)
                            processed_vars.add(var1)
                        if var2 not in cluster_vars:
                            cluster_vars.append(var2)
                            processed_vars.add(var2)
                        break
        
        return clusters
    
    def _interpret_correlation_strength(self, corr_value: float) -> str:
        """
        Interpret correlation strength
        
        Args:
            corr_value: Absolute correlation coefficient
            
        Returns:
            String interpretation of correlation strength
        """
        if corr_value < 0.1:
            return "negligible"
        elif corr_value < 0.3:
            return "weak"
        elif corr_value < 0.5:
            return "moderate"
        elif corr_value < 0.7:
            return "strong"
        else:
            return "very strong"
