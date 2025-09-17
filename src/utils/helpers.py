"""
Helper utilities for Ashta Lakshmi GI Survey Analysis
"""

import logging
import os
from pathlib import Path
from typing import List
import sys

def setup_logging(log_level: str = "INFO", log_file: str = "results/logs/analysis.log") -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def create_directories() -> None:
    """
    Create necessary project directories
    """
    directories = [
        "data/raw",
        "data/processed", 
        "results/figures",
        "results/models",
        "results/reports",
        "results/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_file_exists(file_path: str) -> bool:
    """
    Check if file exists
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()

def get_project_root() -> Path:
    """
    Get project root directory
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        default: Default value if division by zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default

def print_section_header(title: str, width: int = 50) -> None:
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of header line
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")

def save_text_report(content: str, filename: str, directory: str = "results/reports") -> None:
    """
    Save text content to file
    
    Args:
        content: Text content to save
        filename: Name of file
        directory: Directory to save in
    """
    file_path = Path(directory) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Report saved to: {file_path}")

def get_memory_usage() -> str:
    """
    Get current memory usage
    
    Returns:
        Memory usage string
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"{memory_mb:.2f} MB"
    except ImportError:
        return "Memory monitoring not available (install psutil)"

def format_duration(seconds: float) -> str:
    """
    Format duration in human readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.1f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"
