"""
Configuration utilities for Ashta Lakshmi GI Survey Analysis
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set default values if not present
    config = _set_default_config(config)
    
    return config

def _set_default_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set default configuration values
    
    Args:
        config: Input configuration dictionary
        
    Returns:
        Configuration with defaults set
    """
    defaults = {
        'data': {
            'raw_data': 'data/raw/ashta_lakshmi_gi_survey.csv',
            'processed_data': 'data/processed/cleaned_survey_data.csv',
            'output_dir': 'results/'
        },
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'features': {
            'polynomial_degree': 2,
            'feature_selection_k': 10
        },
        'cv': {
            'n_splits': 5,
            'shuffle': True,
            'random_state': 42
        },
        'statistics': {
            'alpha': 0.05,
            'confidence_level': 0.95
        },
        'plots': {
            'figsize': [10, 8],
            'color_palette': 'Set2',
            'dpi': 300
        }
    }
    
    # Merge defaults with provided config
    merged_config = _merge_dicts(defaults, config)
    
    return merged_config

def _merge_dicts(default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge configuration dictionaries
    
    Args:
        default: Default configuration
        custom: Custom configuration
        
    Returns:
        Merged configuration
    """
    result = default.copy()
    
    for key, value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def save_config(config: Dict[str, Any], config_path: str = "config_backup.yaml") -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
