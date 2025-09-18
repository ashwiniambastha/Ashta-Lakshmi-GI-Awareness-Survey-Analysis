"""
Debug helpers for troubleshooting data type issues
"""

import pandas as pd
import numpy as np
import logging

def debug_dataframe_types(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Debug function to print detailed information about dataframe column types
    
    Args:
        df: DataFrame to debug
        name: Name for the dataframe (for logging)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"DEBUG INFO for {name}")
    logger.info(f"{'='*50}")
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    logger.info(f"\nColumn Types:")
    logger.info(f"-" * 30)
    
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        
        # Try to identify problematic columns
        problematic = False
        error_msg = ""
        
        try:
            if dtype == 'object' or dtype.name == 'category':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_numeric_count = numeric_series.isnull().sum() - null_count
                if non_numeric_count > 0:
                    error_msg = f" (contains {non_numeric_count} non-numeric values)"
                    problematic = True
        except Exception as e:
            error_msg = f" (error: {str(e)[:50]})"
            problematic = True
        
        status = "⚠️ " if problematic else "✅ "
        logger.info(f"{status}{col}: {dtype} | Unique: {unique_count} | Nulls: {null_count}{error_msg}")
    
    # Check for mixed types
    logger.info(f"\nPotential Issues:")
    logger.info(f"-" * 30)
    
    object_cols = df.select_dtypes(include=['object']).columns
    category_cols = df.select_dtypes(include=['category']).columns
    
    if len(object_cols) > 0:
        logger.info(f"Object columns (may cause numpy errors): {list(object_cols)}")
    
    if len(category_cols) > 0:
        logger.info(f"Category columns (may cause numpy errors): {list(category_cols)}")
    
    # Sample values from problematic columns
    for col in list(object_cols) + list(category_cols):
        sample_values = df[col].dropna().unique()[:5]
        logger.info(f"Sample values from {col}: {sample_values}")
    
    logger.info(f"{'='*50}\n")

def fix_mixed_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically fix common mixed type issues
    
    Args:
        df: Input dataframe with potential mixed types
        
    Returns:
        DataFrame with fixed types
    """
    logger = logging.getLogger(__name__)
    df_fixed = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.8:  # If >80% are numeric
                    df_fixed[col] = numeric_series
                    logger.info(f"Converted {col} from object to numeric")
                else:
                    # Keep as string
                    df_fixed[col] = df[col].astype(str)
                    logger.info(f"Kept {col} as string")
            except:
                df_fixed[col] = df[col].astype(str)
                
        elif df[col].dtype.name == 'category':
            # Convert category to string to avoid numpy operation errors
            df_fixed[col] = df[col].astype(str)
            logger.info(f"Converted {col} from category to string")
    
    return df_fixed

def validate_for_numpy_operations(X: np.ndarray, operation_name: str = "operation") -> bool:
    """
    Validate if array is suitable for numpy operations
    
    Args:
        X: Array to validate
        operation_name: Name of operation for logging
        
    Returns:
        True if valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if array is numeric
        if X.dtype.kind not in 'biufc':  # boolean, integer, unsigned integer, float, complex
            logger.error(f"Array for {operation_name} has non-numeric dtype: {X.dtype}")
            return False
        
        # Check for infinite values
        if np.isinf(X).any():
            logger.warning(f"Array for {operation_name} contains infinite values")
        
        # Check for NaN values
        if np.isnan(X).any():
            logger.warning(f"Array for {operation_name} contains NaN values")
        
        logger.info(f"Array validation passed for {operation_name}")
        return True
        
    except Exception as e:
        logger.error(f"Array validation failed for {operation_name}: {str(e)}")
        return False
