"""
Data cleaning module for Ashta Lakshmi GI Survey
Handles data preprocessing, cleaning, and transformation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

class DataCleaner:
    """
    Class for comprehensive data cleaning and preprocessing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataCleaner with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main data cleaning pipeline
        
        Args:
            df: Raw dataframe to clean
            
        Returns:
            Cleaned dataframe
        """
        self.logger.info("Starting data cleaning process...")
        
        # Create a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Step 1: Handle duplicates
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Step 2: Standardize categorical values
        cleaned_df = self._standardize_categorical_values(cleaned_df)
        
        # Step 3: Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Step 4: Handle outliers
        cleaned_df = self._handle_outliers(cleaned_df)
        
        # Step 5: Data type conversion
        cleaned_df = self._convert_data_types(cleaned_df)
        
        # Step 6: Feature validation
        cleaned_df = self._validate_cleaned_data(cleaned_df)
        
        self.logger.info(f"Data cleaning completed. Final dataset shape: {cleaned_df.shape}")
        
        return cleaned_df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate records based on Artisan_ID
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with duplicates removed
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(subset=['Artisan_ID'], keep='first')
        removed_count = initial_count - len(df_cleaned)
        
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} duplicate records")
        
        return df_cleaned
    
    def _standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize categorical variable values
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with standardized categorical values
        """
        df_cleaned = df.copy()
        
        # Standardize Yes/No columns
        boolean_columns = ['GI_Aware', 'Received_Subsidy', 'Uses_Ecommerce']
        for col in boolean_columns:
            df_cleaned[col] = df_cleaned[col].str.strip().str.title()
            # Convert to binary for modeling
            df_cleaned[f'{col}_Binary'] = (df_cleaned[col] == 'Yes').astype(int)
        
        # Standardize Gender
        df_cleaned['Gender'] = df_cleaned['Gender'].str.strip().str.title()
        df_cleaned['Gender_Binary'] = (df_cleaned['Gender'] == 'Male').astype(int)
        
        # Standardize State names
        df_cleaned['State'] = df_cleaned['State'].str.strip().str.title()
        
        self.logger.info("Categorical values standardized")
        
        return df_cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate strategies
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        df_cleaned = df.copy()
        missing_summary = df_cleaned.isnull().sum()
        
        if missing_summary.sum() == 0:
            self.logger.info("No missing values found")
            return df_cleaned
        
        self.logger.info(f"Missing values found:\n{missing_summary[missing_summary > 0]}")
        
        # For numerical columns, use KNN imputation
        numerical_cols = ['Age', 'Years_of_Experience']
        if df_cleaned[numerical_cols].isnull().any().any():
            knn_imputer = KNNImputer(n_neighbors=5)
            df_cleaned[numerical_cols] = knn_imputer.fit_transform(df_cleaned[numerical_cols])
            self.logger.info("KNN imputation applied to numerical columns")
        
        # For categorical columns, use mode imputation
        categorical_cols = ['State', 'Gender', 'GI_Aware', 'Received_Subsidy', 'Uses_Ecommerce']
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                mode_value = df_cleaned[col].mode()[0]
                df_cleaned[col].fillna(mode_value, inplace=True)
                self.logger.info(f"Mode imputation applied to {col}")
        
        return df_cleaned
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using Isolation Forest
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with outliers handled
        """
        df_cleaned = df.copy()
        numerical_cols = ['Age', 'Years_of_Experience']
        
        # Use Isolation Forest for outlier detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_labels = iso_forest.fit_predict(df_cleaned[numerical_cols])
        
        # Mark outliers
        df_cleaned['is_outlier'] = (outlier_labels == -1)
        outlier_count = df_cleaned['is_outlier'].sum()
        
        if outlier_count > 0:
            self.logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(df_cleaned)*100:.2f}%)")
            
            # Option 1: Cap outliers instead of removing them
            for col in numerical_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
        
        else:
            self.logger.info("No outliers detected")
        
        return df_cleaned
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert columns to appropriate data types
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with correct data types
        """
        df_cleaned = df.copy()
        
        # Convert to appropriate types
        df_cleaned['Artisan_ID'] = df_cleaned['Artisan_ID'].astype('int64')
        df_cleaned['Age'] = pd.to_numeric(df_cleaned['Age'], errors='coerce').astype('int64')
        df_cleaned['Years_of_Experience'] = pd.to_numeric(df_cleaned['Years_of_Experience'], errors='coerce').astype('int64')
        
        # Keep categorical columns as strings to avoid numpy operation errors
        categorical_cols = ['State', 'Gender', 'GI_Aware', 'Received_Subsidy', 'Uses_Ecommerce']
        for col in categorical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].astype('str')
        
        self.logger.info("Data types converted successfully")
        
        return df_cleaned
    
    def _validate_cleaned_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation of cleaned data
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Validated dataframe
        """
        # Check for any remaining issues
        assert df.isnull().sum().sum() == 0, "Missing values still present after cleaning"
        assert len(df) > 0, "Dataframe is empty after cleaning"
        assert df['Age'].min() >= 0, "Negative age values present"
        assert df['Years_of_Experience'].min() >= 0, "Negative experience values present"
        
        self.logger.info("Data validation completed successfully")
        return df
    
    def save_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Save cleaned data to processed directory
        
        Args:
            df: Cleaned dataframe to save
        """
        output_path = Path(self.config['data']['processed_data'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        self.logger.info(f"Cleaned data saved to: {output_path}")
    
    def get_cleaning_summary(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary of cleaning operations
        
        Args:
            original_df: Original dataframe
            cleaned_df: Cleaned dataframe
            
        Returns:
            Dictionary containing cleaning summary
        """
        summary = {
            'original_records': len(original_df),
            'cleaned_records': len(cleaned_df),
            'records_removed': len(original_df) - len(cleaned_df),
            'removal_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100,
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum(),
            'columns_added': len(cleaned_df.columns) - len(original_df.columns),
            'memory_reduction': original_df.memory_usage(deep=True).sum() - cleaned_df.memory_usage(deep=True).sum()
        }
        
        return summary
