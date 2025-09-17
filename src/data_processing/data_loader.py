"""
Data loading module for Ashta Lakshmi GI Survey
Handles loading and initial validation of raw survey data
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

class DataLoader:
    """
    Class for loading and initial validation of survey data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataLoader with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw survey data from CSV file
        
        Returns:
            DataFrame containing raw survey data
        """
        try:
            data_path = Path(self.config['data']['raw_data'])
            
            if not data_path.exists():
                raise FileNotFoundError(f"Raw data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} records from {data_path}")
            
            # Basic validation
            self._validate_raw_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """
        Validate the structure and content of raw data
        
        Args:
            df: Raw dataframe to validate
        """
        required_columns = [
            'State', 'Artisan_ID', 'Gender', 'GI_Aware', 
            'Received_Subsidy', 'Uses_Ecommerce', 'Age', 'Years_of_Experience'
        ]
        
        # Check if all required columns are present
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for completely empty dataframe
        if len(df) == 0:
            raise ValueError("Dataframe is empty")
        
        # Check for duplicate Artisan_IDs
        duplicate_ids = df['Artisan_ID'].duplicated().sum()
        if duplicate_ids > 0:
            self.logger.warning(f"Found {duplicate_ids} duplicate Artisan_IDs")
        
        # Basic data type validation
        self._validate_data_types(df)
        
        self.logger.info("Raw data validation completed successfully")
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """
        Validate data types and ranges
        
        Args:
            df: DataFrame to validate
        """
        # Check age range
        if df['Age'].min() < 0 or df['Age'].max() > 120:
            self.logger.warning("Age values outside expected range (0-120)")
        
        # Check years of experience
        if df['Years_of_Experience'].min() < 0:
            self.logger.warning("Negative years of experience found")
        
        # Check boolean-like columns
        boolean_columns = ['GI_Aware', 'Received_Subsidy', 'Uses_Ecommerce']
        for col in boolean_columns:
            unique_values = df[col].unique()
            expected_values = ['Yes', 'No']
            unexpected = set(unique_values) - set(expected_values)
            if unexpected:
                self.logger.warning(f"Unexpected values in {col}: {unexpected}")
        
        # Check gender values
        gender_values = df['Gender'].unique()
        expected_genders = ['Male', 'Female']
        unexpected_genders = set(gender_values) - set(expected_genders)
        if unexpected_genders:
            self.logger.warning(f"Unexpected gender values: {unexpected_genders}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the loaded data
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_records': df.duplicated().sum(),
            'states_covered': df['State'].nunique(),
            'state_distribution': df['State'].value_counts().to_dict()
        }
        
        return summary
    
    def load_external_data(self, file_path: str) -> pd.DataFrame:
        """
        Load external data files for enrichment
        
        Args:
            file_path: Path to external data file
            
        Returns:
            DataFrame containing external data
        """
        try:
            ext_data = pd.read_csv(file_path)
            self.logger.info(f"Loaded external data: {file_path}")
            return ext_data
        except Exception as e:
            self.logger.error(f"Error loading external data: {str(e)}")
            raise
