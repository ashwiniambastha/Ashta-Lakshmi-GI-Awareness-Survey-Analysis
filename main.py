"""
Main execution script for Ashta Lakshmi GI Survey Analysis
Run complete data science pipeline from data loading to model evaluation
"""

import os
import yaml
import logging
from pathlib import Path

# Import custom modules
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.eda.statistical_analysis import StatisticalAnalyzer
from src.eda.correlation_analysis import CorrelationAnalyzer
from src.visualization.eda_plots import EDAVisualizer
from src.feature_engineering.feature_creation import FeatureEngineer
from src.feature_engineering.feature_selection import FeatureSelector
from src.modeling.ensemble_models import EnsembleModeler
from src.modeling.model_evaluation import ModelEvaluator
from src.utils.config import load_config
from src.utils.helpers import setup_logging, create_directories

def main():
    """
    Execute complete data science pipeline
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Ashta Lakshmi GI Survey Analysis")
    
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    create_directories()
    
    try:
        # Step 1: Data Loading and Cleaning
        logger.info("Step 1: Loading and cleaning data...")
        loader = DataLoader(config)
        raw_data = loader.load_raw_data()
        
        cleaner = DataCleaner(config)
        clean_data = cleaner.clean_data(raw_data)
        cleaner.save_cleaned_data(clean_data)
        
        # Step 2: Exploratory Data Analysis
        logger.info("Step 2: Performing exploratory data analysis...")
        stat_analyzer = StatisticalAnalyzer(config)
        stat_results = stat_analyzer.perform_analysis(clean_data)
        
        corr_analyzer = CorrelationAnalyzer(config)
        corr_results = corr_analyzer.analyze_correlations(clean_data)
        
        # Step 3: Data Visualization
        logger.info("Step 3: Creating visualizations...")
        visualizer = EDAVisualizer(config)
        visualizer.create_all_plots(clean_data, stat_results, corr_results)
        
        # Step 4: Feature Engineering
        logger.info("Step 4: Engineering features...")
        feature_engineer = FeatureEngineer(config)
        engineered_data = feature_engineer.create_features(clean_data)
        
        feature_selector = FeatureSelector(config)
        selected_features = feature_selector.select_features(engineered_data)
        
        # Step 5: Model Training and Evaluation
        logger.info("Step 5: Training and evaluating models...")
        modeler = EnsembleModeler(config)
        trained_models = modeler.train_models(selected_features)
        
        evaluator = ModelEvaluator(config)
        evaluation_results = evaluator.evaluate_models(trained_models, selected_features)
        
        # Step 6: Generate Final Report
        logger.info("Step 6: Generating final report...")
        generate_final_report(stat_results, corr_results, evaluation_results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

def generate_final_report(stat_results, corr_results, evaluation_results):
    """
    Generate comprehensive final report
    """
    report_path = Path("results/reports/final_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("ASHTA LAKSHMI GI SURVEY - FINAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("STATISTICAL ANALYSIS RESULTS:\n")
        f.write("-" * 30 + "\n")
        for key, value in stat_results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nCORRELATION ANALYSIS RESULTS:\n")
        f.write("-" * 30 + "\n")
        for key, value in corr_results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nMODEL EVALUATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        for key, value in evaluation_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Final report saved to: {report_path}")

if __name__ == "__main__":
    main()
