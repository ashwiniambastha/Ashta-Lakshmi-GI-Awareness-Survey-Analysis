# src/__init__.py
"""
Ashta Lakshmi GI Survey Analysis Package
A comprehensive data science project for analyzing GI awareness among Northeast Indian artisans
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

# src/data_processing/__init__.py
"""
Data processing modules for loading and cleaning survey data
"""

from .data_loader import DataLoader
from .data_cleaner import DataCleaner

__all__ = ['DataLoader', 'DataCleaner']

# src/eda/__init__.py
"""
Exploratory Data Analysis modules
"""

from .statistical_analysis import StatisticalAnalyzer
from .correlation_analysis import CorrelationAnalyzer

__all__ = ['StatisticalAnalyzer', 'CorrelationAnalyzer']

# src/visualization/__init__.py
"""
Visualization modules for creating plots and dashboards
"""

from .eda_plots import EDAVisualizer

__all__ = ['EDAVisualizer']

# src/feature_engineering/__init__.py
"""
Feature engineering modules for creating and selecting features
"""

from .feature_creation import FeatureEngineer
from .feature_selection import FeatureSelector

__all__ = ['FeatureEngineer', 'FeatureSelector']

# src/modeling/__init__.py
"""
Machine learning modeling modules
"""

from .ensemble_models import EnsembleModeler
from .model_evaluation import ModelEvaluator

__all__ = ['EnsembleModeler', 'ModelEvaluator']

# src/utils/__init__.py
"""
Utility modules for configuration and helper functions
"""

from .config import load_config, save_config
from .helpers import setup_logging, create_directories

__all__ = ['load_config', 'save_config', 'setup_logging', 'create_directories']
