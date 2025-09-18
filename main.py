"""
Main execution script for Ashta Lakshmi GI Survey Analysis
Run complete data science pipeline with detailed step-by-step output
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

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

def print_step_header(step_num: int, title: str, emoji: str = "📊"):
    """Print formatted step header"""
    print(f"\n{emoji} STEP {step_num}: {title.upper()}")
    print("=" * 80)

def print_section(title: str, emoji: str = "📊"):
    """Print formatted section header"""
    print(f"\n{emoji} {title.upper()}")
    print("=" * 60)

def print_subsection(title: str, emoji: str = "•"):
    """Print formatted subsection"""
    print(f"\n{emoji} {title}:")

def print_success(message: str):
    """Print success message"""
    print(f"   ✅ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"   • {message}")

def print_warning(message: str):
    """Print warning message"""
    print(f"   ⚠️  {message}")

def print_error(message: str):
    """Print error message"""
    print(f"   ❌ {message}")

def analyze_data_overview(df: pd.DataFrame) -> dict:
    """Generate comprehensive data overview"""
    overview = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024,  # KB
        'dtypes': df.dtypes.value_counts().to_dict(),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns)
    }
    return overview

def print_data_overview(df: pd.DataFrame, name: str):
    """Print detailed data overview"""
    overview = analyze_data_overview(df)
    
    print_subsection(f"{name} Overview")
    print_info(f"Shape: {overview['shape'][0]} records × {overview['shape'][1]} features")
    print_info(f"Missing Values: {overview['missing_values']} total")
    print_info(f"Memory Usage: {overview['memory_usage']:.2f} KB")
    
    dtype_str = ', '.join([f"{str(k)}: {v}" for k, v in overview['dtypes'].items()])
    print_info(f"Data Types: {{{dtype_str}}}")
    print_info(f"Numerical Columns: {overview['numeric_cols']}")
    print_info(f"Categorical Columns: {overview['categorical_cols']}")
    
    # Missing values breakdown
    missing_by_col = df.isnull().sum()
    missing_cols = missing_by_col[missing_by_col > 0]
    if len(missing_cols) > 0:
        print_info("Columns with Missing Values:")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"     - {col}: {pct:.1f}% missing")

def print_statistical_insights(stat_results: dict):
    """Print key statistical insights"""
    print_section("Key Statistical Insights", "🔍")
    
    if 'descriptive_stats' in stat_results:
        desc_stats = stat_results['descriptive_stats']
        
        # Numerical variables insights
        for var in ['Age', 'Years_of_Experience']:
            if var in desc_stats:
                stats = desc_stats[var]
                print_info(f"{var}: Mean={stats['mean']:.1f}, Median={stats['median']:.1f}, Std={stats['std']:.1f}")
        
        # Categorical variables insights
        for var in ['GI_Aware', 'Uses_Ecommerce', 'Received_Subsidy']:
            if var in desc_stats:
                stats = desc_stats[var]
                if 'percentages' in stats:
                    yes_pct = stats['percentages'].get('Yes', 0)
                    print_info(f"{var.replace('_', ' ')} Rate: {yes_pct:.1f}%")
    
    # Chi-square test results
    if 'chi_square_tests' in stat_results:
        print_subsection("Significant Associations", "🔗")
        chi_tests = stat_results['chi_square_tests']
        
        for test_name, results in chi_tests.items():
            if results.get('significant', False):
                vars_tested = test_name.replace('_vs_', ' vs ')
                print_info(f"{vars_tested}: SIGNIFICANT (p={results['p_value']:.4f})")

def print_model_results(model_results: dict):
    """Print model performance results"""
    print_section("Model Performance Results", "🤖")
    
    best_models = model_results.get('best_models', {})
    
    if best_models:
        print_subsection("Best Performing Models")
        
        for model_type, (model_name, model_info) in best_models.items():
            results = model_info.get('results', {})
            accuracy = results.get('accuracy', 0)
            f1_score = results.get('f1_score', 0)
            
            print_info(f"{model_type.replace('_', ' ').title()}: {model_name}")
            print(f"     - Accuracy: {accuracy:.1%}")
            print(f"     - F1-Score: {f1_score:.3f}")
    
    # All models comparison
    all_models = model_results.get('models', {})
    if all_models:
        print_subsection("All Models Comparison")
        
        for model_name, model_info in all_models.items():
            if 'results' in model_info:
                results = model_info['results']
                accuracy = results.get('accuracy', 0)
                f1_score = results.get('f1_score', 0)
                print_info(f"{model_name.replace('_', ' ').title()}: Acc={accuracy:.1%}, F1={f1_score:.3f}")

def calculate_business_impact(df: pd.DataFrame) -> dict:
    """Calculate business impact metrics"""
    total_artisans = len(df)
    
    # GI Awareness metrics
    gi_aware_count = df['GI_Aware_Binary'].sum() if 'GI_Aware_Binary' in df.columns else 0
    gi_awareness_rate = (gi_aware_count / total_artisans) * 100 if total_artisans > 0 else 0
    
    # E-commerce adoption
    ecom_count = df['Uses_Ecommerce_Binary'].sum() if 'Uses_Ecommerce_Binary' in df.columns else 0
    ecom_rate = (ecom_count / total_artisans) * 100 if total_artisans > 0 else 0
    
    # Subsidy recipients
    subsidy_count = df['Received_Subsidy_Binary'].sum() if 'Received_Subsidy_Binary' in df.columns else 0
    subsidy_rate = (subsidy_count / total_artisans) * 100 if total_artisans > 0 else 0
    
    # Gender analysis
    gender_gap = 0
    if 'Gender' in df.columns and 'GI_Aware_Binary' in df.columns:
        male_awareness = df[df['Gender'] == 'Male']['GI_Aware_Binary'].mean()
        female_awareness = df[df['Gender'] == 'Female']['GI_Aware_Binary'].mean()
        gender_gap = abs(male_awareness - female_awareness) * 100
    
    return {
        'total_artisans': total_artisans,
        'gi_awareness_rate': gi_awareness_rate,
        'ecommerce_rate': ecom_rate,
        'subsidy_rate': subsidy_rate,
        'gender_gap': gender_gap
    }

def print_business_impact(df: pd.DataFrame):
    """Print business impact analysis"""
    print_section("Business Impact Analysis", "💼")
    
    impact = calculate_business_impact(df)
    
    print_subsection("Key Metrics")
    print_info(f"Total Artisans Surveyed: {impact['total_artisans']}")
    print_info(f"GI Awareness Rate: {impact['gi_awareness_rate']:.1f}%")
    print_info(f"E-commerce Adoption Rate: {impact['ecommerce_rate']:.1f}%")
    print_info(f"Government Subsidy Recipients: {impact['subsidy_rate']:.1f}%")
    
    if impact['gender_gap'] > 0:
        print_info(f"Gender Awareness Gap: {impact['gender_gap']:.1f} percentage points")
    
    print_subsection("Policy Recommendations", "🎯")
    if impact['gi_awareness_rate'] < 70:
        print_info("PRIORITY: Increase GI awareness campaigns")
    if impact['ecommerce_rate'] < 30:
        print_info("PRIORITY: Digital literacy programs needed")
    if impact['gender_gap'] > 10:
        print_info("PRIORITY: Address gender-based awareness gap")

def main():
    """
    Execute complete data science pipeline with detailed output
    """
    start_time = time.time()
    
    print("🌟 ASHTA LAKSHMI GI SURVEY - COMPREHENSIVE DATA SCIENCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    print_section("Setting Up Analysis Environment", "⚙️")
    config = load_config()
    print_success("Configuration loaded successfully")
    
    # Create necessary directories
    create_directories()
    print_success("Project directories initialized")
    
    try:
        # Step 1: Data Loading and Initial Exploration
        print_step_header(1, "Data Loading & Initial Exploration", "📊")
        
        loader = DataLoader(config)
        raw_data = loader.load_raw_data()
        print_success(f"Raw data loaded: {raw_data.shape[0]} records")
        
        print_data_overview(raw_data, "Raw Dataset")
        
        # Step 2: Data Cleaning and Preprocessing
        print_step_header(2, "Data Cleaning & Preprocessing", "🧹")
        
        cleaner = DataCleaner(config)
        clean_data = cleaner.clean_data(raw_data)
        print_success("Data cleaning completed")
        
        # Show cleaning impact
        original_nulls = raw_data.isnull().sum().sum()
        cleaned_nulls = clean_data.isnull().sum().sum()
        print_info(f"Missing values: {original_nulls} → {cleaned_nulls}")
        print_info(f"New features added: {clean_data.shape[1] - raw_data.shape[1]}")
        
        print_data_overview(clean_data, "Cleaned Dataset")
        
        # Step 3: Statistical Analysis
        print_step_header(3, "Statistical Analysis", "📈")
        
        stat_analyzer = StatisticalAnalyzer(config)
        stat_results = stat_analyzer.perform_analysis(clean_data)
        print_success("Statistical analysis completed")
        
        print_statistical_insights(stat_results)
        
        # Step 4: Feature Engineering
        print_step_header(4, "Advanced Feature Engineering", "🔧")
        
        feature_engineer = FeatureEngineer(config)
        engineered_data = feature_engineer.create_features(clean_data)
        print_success("Feature engineering completed")
        
        original_features = clean_data.shape[1]
        new_features = engineered_data.shape[1]
        print_info(f"Features: {original_features} → {new_features} ({new_features - original_features} new)")
        
        # Show key engineered features
        new_feature_cols = set(engineered_data.columns) - set(clean_data.columns)
        key_features = [col for col in new_feature_cols if any(keyword in col.lower() 
                       for keyword in ['ratio', 'interaction', 'score', 'deviation'])][:5]
        if key_features:
            print_info("Key engineered features:")
            for feat in key_features:
                print(f"     - {feat}")
        
        # Step 5: Feature Selection
        print_step_header(5, "Intelligent Feature Selection", "🎯")
        
        feature_selector = FeatureSelector(config)
        selected_features = feature_selector.select_features(engineered_data)
        print_success("Feature selection completed")
        
        selected_count = len([col for col in selected_features.columns 
                            if col not in ['Artisan_ID', 'State', 'Gender', 'GI_Aware_Binary']])
        print_info(f"Selected {selected_count} optimal features from {new_features} candidates")
        
        # Step 6: Machine Learning Models
        print_step_header(6, "Advanced Machine Learning", "🤖")
        
        modeler = EnsembleModeler(config)
        trained_models = modeler.train_models(selected_features)
        print_success("Model training completed")
        
        print_model_results(trained_models)
        
        # Step 7: Model Evaluation
        print_step_header(7, "Comprehensive Model Evaluation", "📊")
        
        evaluator = ModelEvaluator(config)
        evaluation_results = evaluator.evaluate_models(trained_models, selected_features)
        print_success("Model evaluation completed")
        
        # Print evaluation insights
        best_recommendations = evaluation_results.get('best_model_recommendations', {})
        if best_recommendations:
            print_subsection("Model Recommendations")
            for criterion, model_name in best_recommendations.items():
                print_info(f"{criterion.replace('_', ' ').title()}: {model_name.replace('_', ' ')}")
        
        # Step 8: Business Impact Analysis
        print_step_header(8, "Business Impact & Insights", "💼")
        
        print_business_impact(clean_data)
        
        # Step 9: Visualization Generation
        print_step_header(9, "Creating Visualizations", "📊")
        
        visualizer = EDAVisualizer(config)
        visualizer.create_all_plots(clean_data, stat_results, 
                                   CorrelationAnalyzer(config).analyze_correlations(clean_data))
        print_success("Visualizations created and saved")
        print_info("Check 'results/figures/' directory for all plots")
        
        # Step 10: Final Report Generation
        print_step_header(10, "Generating Final Report", "📄")
        
        generate_executive_summary(clean_data, stat_results, trained_models, evaluation_results)
        print_success("Executive summary generated")
        
        # Final Summary
        elapsed_time = time.time() - start_time
        print_section("Analysis Complete", "✅")
        print_success(f"Total execution time: {elapsed_time:.1f} seconds")
        print_success(f"Results saved in 'results/' directory")
        print_success(f"Visualizations saved in 'results/figures/' directory")
        
        print(f"\n🎉 ASHTA LAKSHMI GI SURVEY ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Key Findings: Check executive summary in results/reports/")
        print(f"📈 Visualizations: Check results/figures/ directory")
        print(f"🤖 Models: Check results/models/ directory")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"Analysis failed after {elapsed_time:.1f} seconds")
        print_error(f"Error: {str(e)}")
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

def generate_executive_summary(clean_data, stat_results, model_results, evaluation_results):
    """Generate executive summary report"""
    summary_path = Path("results/reports/executive_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    impact = calculate_business_impact(clean_data)
    
    with open(summary_path, 'w') as f:
        f.write("ASHTA LAKSHMI GI SURVEY - EXECUTIVE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Artisans Surveyed: {impact['total_artisans']}\n")
        f.write(f"GI Awareness Rate: {impact['gi_awareness_rate']:.1f}%\n")
        f.write(f"E-commerce Adoption: {impact['ecommerce_rate']:.1f}%\n")
        f.write(f"Subsidy Recipients: {impact['subsidy_rate']:.1f}%\n")
        f.write(f"Gender Awareness Gap: {impact['gender_gap']:.1f} percentage points\n\n")
        
        f.write("MACHINE LEARNING RESULTS:\n")
        f.write("-" * 30 + "\n")
        best_models = model_results.get('best_models', {})
        for model_type, (model_name, model_info) in best_models.items():
            results = model_info.get('results', {})
            accuracy = results.get('accuracy', 0)
            f.write(f"{model_type}: {model_name} (Accuracy: {accuracy:.1%})\n")
        
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()
