"""
Statistical analysis module for Ashta Lakshmi GI Survey
Performs comprehensive statistical tests and hypothesis testing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Class for performing statistical analysis on survey data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StatisticalAnalyzer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.alpha = config.get('statistics', {}).get('alpha', 0.05)
        
    def perform_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dictionary containing all statistical results
        """
        self.logger.info("Starting statistical analysis...")
        
        results = {}
        
        # Descriptive statistics
        results['descriptive_stats'] = self._descriptive_statistics(df)
        
        # Chi-square tests for categorical associations
        results['chi_square_tests'] = self._chi_square_tests(df)
        
        # Correlation analysis
        results['correlation_analysis'] = self._correlation_analysis(df)
        
        # Group comparisons
        results['group_comparisons'] = self._group_comparisons(df)
        
        # Hypothesis tests
        results['hypothesis_tests'] = self._hypothesis_tests(df)
        
        # Confidence intervals
        results['confidence_intervals'] = self._confidence_intervals(df)
        
        self.logger.info("Statistical analysis completed")
        return results
    
    def _descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing descriptive statistics
        """
        desc_stats = {}
        
        # Numerical variables
        numerical_cols = ['Age', 'Years_of_Experience']
        for col in numerical_cols:
            desc_stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'q25': df[col].quantile(0.25),
                'q75': df[col].quantile(0.75)
            }
        
        # Categorical variables
        categorical_cols = ['State', 'Gender', 'GI_Aware', 'Received_Subsidy', 'Uses_Ecommerce']
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            percentages = df[col].value_counts(normalize=True) * 100
            
            desc_stats[col] = {
                'unique_values': df[col].nunique(),
                'mode': df[col].mode()[0],
                'value_counts': value_counts.to_dict(),
                'percentages': percentages.to_dict()
            }
        
        return desc_stats
    
    def _chi_square_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform chi-square tests for categorical associations
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing chi-square test results
        """
        chi_square_results = {}
        
        # Test associations between key variables
        test_pairs = [
            ('Gender', 'GI_Aware'),
            ('State', 'GI_Aware'),
            ('Gender', 'Uses_Ecommerce'),
            ('State', 'Uses_Ecommerce'),
            ('GI_Aware', 'Received_Subsidy'),
            ('Uses_Ecommerce', 'Received_Subsidy')
        ]
        
        for var1, var2 in test_pairs:
            try:
                # Create contingency table
                contingency_table = pd.crosstab(df[var1], df[var2])
                
                # Perform chi-square test
                chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                
                # Calculate effect size (Cramér's V)
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                
                chi_square_results[f'{var1}_vs_{var2}'] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'cramers_v': cramers_v,
                    'significant': p_value < self.alpha,
                    'contingency_table': contingency_table.to_dict()
                }
                
            except Exception as e:
                self.logger.warning(f"Could not perform chi-square test for {var1} vs {var2}: {str(e)}")
        
        return chi_square_results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform correlation analysis for numerical variables
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing correlation results
        """
        correlation_results = {}
        
        # Numerical columns for correlation
        numerical_cols = ['Age', 'Years_of_Experience']
        binary_cols = ['GI_Aware_Binary', 'Received_Subsidy_Binary', 'Uses_Ecommerce_Binary']
        all_corr_cols = numerical_cols + binary_cols
        
        # Pearson correlation
        pearson_corr = df[all_corr_cols].corr(method='pearson')
        correlation_results['pearson_correlation'] = pearson_corr.to_dict()
        
        # Spearman correlation (rank-based)
        spearman_corr = df[all_corr_cols].corr(method='spearman')
        correlation_results['spearman_correlation'] = spearman_corr.to_dict()
        
        # Significant correlations
        significant_correlations = []
        for i in range(len(all_corr_cols)):
            for j in range(i+1, len(all_corr_cols)):
                col1, col2 = all_corr_cols[i], all_corr_cols[j]
                corr_coef, p_value = pearsonr(df[col1], df[col2])
                
                if p_value < self.alpha:
                    significant_correlations.append({
                        'variable1': col1,
                        'variable2': col2,
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'strength': self._interpret_correlation_strength(abs(corr_coef))
                    })
        
        correlation_results['significant_correlations'] = significant_correlations
        
        return correlation_results
    
    def _group_comparisons(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform group comparisons using t-tests and ANOVA
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing group comparison results
        """
        group_results = {}
        
        # T-tests for binary groupings
        binary_groupings = [
            ('Gender', 'Age'),
            ('Gender', 'Years_of_Experience'),
            ('GI_Aware', 'Age'),
            ('GI_Aware', 'Years_of_Experience'),
            ('Uses_Ecommerce', 'Age'),
            ('Uses_Ecommerce', 'Years_of_Experience')
        ]
        
        for group_var, numeric_var in binary_groupings:
            try:
                groups = df.groupby(group_var)[numeric_var].apply(list)
                if len(groups) == 2:
                    group1, group2 = groups.iloc[0], groups.iloc[1]
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    
                    group_results[f'{group_var}_vs_{numeric_var}_ttest'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.alpha,
                        'group1_mean': np.mean(group1),
                        'group2_mean': np.mean(group2),
                        'effect_size': self._calculate_cohens_d(group1, group2)
                    }
            except Exception as e:
                self.logger.warning(f"Could not perform t-test for {group_var} vs {numeric_var}: {str(e)}")
        
        # ANOVA for multi-group comparisons (State-wise)
        for numeric_var in ['Age', 'Years_of_Experience']:
            try:
                state_groups = [group[numeric_var].values for name, group in df.groupby('State')]
                f_stat, p_value = stats.f_oneway(*state_groups)
                
                group_results[f'State_vs_{numeric_var}_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
            except Exception as e:
                self.logger.warning(f"Could not perform ANOVA for State vs {numeric_var}: {str(e)}")
        
        return group_results
    
    def _hypothesis_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform specific hypothesis tests based on research questions
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing hypothesis test results
        """
        hypothesis_results = {}
        
        # Hypothesis 1: GI awareness differs by gender
        male_gi_aware = df[df['Gender'] == 'Male']['GI_Aware_Binary']
        female_gi_aware = df[df['Gender'] == 'Female']['GI_Aware_Binary']
        
        # Proportion test
        male_prop = male_gi_aware.mean()
        female_prop = female_gi_aware.mean()
        
        hypothesis_results['gender_gi_awareness'] = {
            'male_awareness_rate': male_prop,
            'female_awareness_rate': female_prop,
            'difference': male_prop - female_prop,
            'percentage_gap': abs(male_prop - female_prop) * 100
        }
        
        # Hypothesis 2: E-commerce usage correlates with GI awareness
        ecommerce_gi_corr, p_val = pearsonr(df['Uses_Ecommerce_Binary'], df['GI_Aware_Binary'])
        
        hypothesis_results['ecommerce_gi_correlation'] = {
            'correlation': ecommerce_gi_corr,
            'p_value': p_val,
            'significant': p_val < self.alpha
        }
        
        return hypothesis_results
    
    def _confidence_intervals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate confidence intervals for key metrics
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary containing confidence intervals
        """
        confidence_level = self.config.get('statistics', {}).get('confidence_level', 0.95)
        alpha_ci = 1 - confidence_level
        
        ci_results = {}
        
        # Confidence intervals for proportions
        binary_vars = ['GI_Aware_Binary', 'Received_Subsidy_Binary', 'Uses_Ecommerce_Binary']
        
        for var in binary_vars:
            prop = df[var].mean()
            n = len(df)
            se = np.sqrt(prop * (1 - prop) / n)
            z_score = stats.norm.ppf(1 - alpha_ci/2)
            
            ci_lower = prop - z_score * se
            ci_upper = prop + z_score * se
            
            ci_results[f'{var}_proportion'] = {
                'estimate': prop,
                'confidence_interval': [ci_lower, ci_upper],
                'margin_of_error': z_score * se
            }
        
        # Confidence intervals for means
        numerical_vars = ['Age', 'Years_of_Experience']
        
        for var in numerical_vars:
            mean_val = df[var].mean()
            std_val = df[var].std()
            n = len(df)
            se = std_val / np.sqrt(n)
            t_score = stats.t.ppf(1 - alpha_ci/2, n-1)
            
            ci_lower = mean_val - t_score * se
            ci_upper = mean_val + t_score * se
            
            ci_results[f'{var}_mean'] = {
                'estimate': mean_val,
                'confidence_interval': [ci_lower, ci_upper],
                'margin_of_error': t_score * se
            }
        
        return ci_results
    
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
    
    def _calculate_cohens_d(self, group1: list, group2: list) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            group1: First group values
            group2: Second group values
            
        Returns:
            Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        return cohens_d
