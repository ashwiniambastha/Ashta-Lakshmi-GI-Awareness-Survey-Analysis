"""
Unit tests for Ashta Lakshmi GI Survey Analysis models
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.feature_engineering.feature_creation import FeatureEngineer
from src.modeling.ensemble_models import EnsembleModeler
from src.utils.config import load_config

class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality"""
    
    def setUp(self):
        """Set up test data and configuration"""
        self.config = {
            'data': {
                'raw_data': 'data/raw/test_data.csv',
                'processed_data': 'data/processed/test_cleaned.csv',
                'output_dir': 'test_results/'
            },
            'features': {
                'polynomial_degree': 2,
                'feature_selection_k': 5
            },
            'models': {
                'random_forest': {'n_estimators': 10, 'random_state': 42}
            }
        }
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'State': ['Assam', 'Manipur', 'Sikkim'] * 10,
            'Artisan_ID': range(1001, 1031),
            'Gender': ['Male', 'Female'] * 15,
            'GI_Aware': ['Yes', 'No'] * 15,
            'Received_Subsidy': ['No', 'Yes'] * 15,
            'Uses_Ecommerce': ['No', 'No', 'Yes'] * 10,
            'Age': np.random.randint(18, 65, 30),
            'Years_of_Experience': np.random.randint(1, 40, 30)
        })
    
    def test_data_loader(self):
        """Test data loading functionality"""
        loader = DataLoader(self.config)
        
        # Test data validation
        try:
            loader._validate_raw_data(self.sample_data)
            self.assertTrue(True, "Data validation passed")
        except Exception as e:
            self.fail(f"Data validation failed: {str(e)}")
    
    def test_data_cleaner(self):
        """Test data cleaning functionality"""
        cleaner = DataCleaner(self.config)
        
        # Test cleaning pipeline
        cleaned_data = cleaner.clean_data(self.sample_data)
        
        # Check that binary columns were created
        self.assertIn('GI_Aware_Binary', cleaned_data.columns)
        self.assertIn('Gender_Binary', cleaned_data.columns)
        
        # Check no missing values
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.config = {
            'features': {'polynomial_degree': 2}
        }
        
        self.sample_data = pd.DataFrame({
            'Age': [25, 35, 45, 55],
            'Years_of_Experience': [5, 15, 25, 35],
            'GI_Aware_Binary': [1, 0, 1, 0],
            'Gender_Binary': [1, 0, 1, 0],
            'State': ['Assam', 'Manipur', 'Sikkim', 'Nagaland']
        })
    
    def test_feature_creation(self):
        """Test feature creation"""
        engineer = FeatureEngineer(self.config)
        
        # Test derived features
        featured_data = engineer._create_derived_features(self.sample_data)
        
        # Check that new features were created
        self.assertIn('Age_Experience_Ratio', featured_data.columns)
        self.assertIn('Career_Start_Age', featured_data.columns)
        
        # Check values are reasonable
        self.assertTrue(all(featured_data['Career_Start_Age'] >= 0))

class TestModeling(unittest.TestCase):
    """Test modeling functionality"""
    
    def setUp(self):
        """Set up test data for modeling"""
        self.config = {
            'models': {
                'random_forest': {'n_estimators': 10, 'random_state': 42}
            },
            'data': {'output_dir': 'test_results/'}
        }
        
        # Create larger sample for modeling
        np.random.seed(42)
        n_samples = 100
        
        self.model_data = pd.DataFrame({
            'Age': np.random.randint(18, 65, n_samples),
            'Years_of_Experience': np.random.randint(1, 40, n_samples),
            'Gender_Binary': np.random.binomial(1, 0.5, n_samples),
            'State_Encoded': np.random.randint(0, 8, n_samples),
            'GI_Aware_Binary': np.random.binomial(1, 0.6, n_samples),
            'Received_Subsidy_Binary': np.random.binomial(1, 0.3, n_samples),
            'Uses_Ecommerce_Binary': np.random.binomial(1, 0.2, n_samples),
            'Artisan_ID': range(1001, 1001 + n_samples)
        })
    
    def test_model_training(self):
        """Test model training functionality"""
        modeler = EnsembleModeler(self.config)
        
        # Test data preparation
        X_train, X_test, y_train, y_test = modeler._prepare_modeling_data(self.model_data)
        
        # Check data shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(self.model_data))
        self.assertEqual(len(y_train), X_train.shape[0])
        self.assertEqual(len(y_test), X_test.shape[0])
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        from sklearn.ensemble import RandomForestClassifier
        
        modeler = EnsembleModeler(self.config)
        
        # Create simple model for testing
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = self.model_data[['Age', 'Years_of_Experience', 'Gender_Binary']].values
        y = self.model_data['GI_Aware_Binary'].values
        
        model.fit(X, y)
        
        # Test evaluation
        results = modeler._evaluate_model(model, X, y, 'Test Model')
        
        # Check that all metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, results)
            self.assertIsInstance(results[metric], float)
            self.assertTrue(0 <= results[metric] <= 1)

if __name__ == '__main__':
    unittest.main()
