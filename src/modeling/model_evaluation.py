"""
Model evaluation module for Ashta Lakshmi GI Survey
Comprehensive evaluation of trained models with advanced metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

class ModelEvaluator:
    """
    Class for comprehensive model evaluation and performance analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ModelEvaluator with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
    def evaluate_models(self, trained_models: Dict[str, Any], 
                       selected_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate all trained models comprehensively
        
        Args:
            trained_models: Dictionary containing trained models
            selected_features: DataFrame with selected features
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        # Extract test data
        if 'test_data' in trained_models:
            X_test, y_test = trained_models['test_data']
        else:
            # Fallback: create test split
            X_test, y_test = self._create_test_split(selected_features)
        
        # Evaluate all models
        for model_name, model_info in trained_models['models'].items():
            if 'model' in model_info:
                model = model_info['model']
                evaluation = self._comprehensive_evaluation(model, X_test, y_test, model_name)
                self.evaluation_results[model_name] = evaluation
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report()
        
        # Create visualizations
        self._create_evaluation_plots(X_test, y_test)
        
        # Save results
        self._save_evaluation_results()
        
        self.logger.info("Model evaluation completed")
        return {
            'individual_results': self.evaluation_results,
            'comparison_report': comparison_report,
            'best_model_recommendations': self._recommend_best_models()
        }
    
    def _comprehensive_evaluation(self, model, X_test: np.ndarray, 
                                y_test: np.ndarray, model_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a single model
        
        Args:
            model: Trained model
            X_test, y_test: Test data
            model_name: Name of the model
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        results = {}
        
        # Basic predictions
        y_pred = model.predict(X_test)
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted')
        results['recall'] = recall_score(y_test, y_pred, average='weighted')
        results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = class_report
        
        # ROC AUC and curves (if model supports probability prediction)
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                results['roc_auc'] = roc_auc_score(y_test, y_proba)
                
                # ROC curve data
                fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # Precision-Recall curve
                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
                results['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
                
            except Exception as e:
                self.logger.warning(f"Could not compute ROC metrics for {model_name}: {str(e)}")
                results['roc_auc'] = None
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_.tolist()
        elif hasattr(model, 'coef_'):
            results['feature_importance'] = np.abs(model.coef_[0]).tolist()
        
        # Cross-validation scores
        try:
            # Use a subset for faster CV evaluation
            X_cv = X_test[:min(100, len(X_test))]
            y_cv = y_test[:min(100, len(y_test))]
            
            cv_scores = cross_val_score(model, X_cv, y_cv, cv=3, scoring='accuracy')
            results['cross_val_mean'] = cv_scores.mean()
            results['cross_val_std'] = cv_scores.std()
            results['cross_val_scores'] = cv_scores.tolist()
            
        except Exception as e:
            self.logger.warning(f"Could not compute CV scores for {model_name}: {str(e)}")
        
        return results
    
    def _create_test_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create test split if not provided
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of test data (X_test, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        target_col = 'GI_Aware_Binary'
        exclude_cols = [target_col, 'Artisan_ID', 'State', 'Gender']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        return X_test, y_test
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comparison report across all models
        
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Collect metrics for comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics:
            comparison[metric] = {}
            values = []
            
            for model_name, results in self.evaluation_results.items():
                if metric in results and results[metric] is not None:
                    comparison[metric][model_name] = results[metric]
                    values.append(results[metric])
            
            if values:
                comparison[f'{metric}_best'] = max(comparison[metric].items(), key=lambda x: x[1])
                comparison[f'{metric}_avg'] = np.mean(values)
                comparison[f'{metric}_std'] = np.std(values)
        
        return comparison
    
    def _recommend_best_models(self) -> Dict[str, str]:
        """
        Recommend best models for different criteria
        
        Returns:
            Dictionary containing best model recommendations
        """
        recommendations = {}
        
        # Best accuracy
        best_accuracy = max(self.evaluation_results.items(), 
                           key=lambda x: x[1].get('accuracy', 0))
        recommendations['best_accuracy'] = best_accuracy[0]
        
        # Best F1 score
        best_f1 = max(self.evaluation_results.items(), 
                     key=lambda x: x[1].get('f1_score', 0))
        recommendations['best_f1'] = best_f1[0]
        
        # Best ROC AUC (if available)
        roc_models = {k: v for k, v in self.evaluation_results.items() 
                     if v.get('roc_auc') is not None}
        if roc_models:
            best_roc = max(roc_models.items(), 
                          key=lambda x: x[1].get('roc_auc', 0))
            recommendations['best_roc_auc'] = best_roc[0]
        
        # Most balanced (high accuracy + high F1)
        balanced_scores = {}
        for name, results in self.evaluation_results.items():
            acc = results.get('accuracy', 0)
            f1 = results.get('f1_score', 0)
            balanced_scores[name] = (acc + f1) / 2
        
        best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
        recommendations['most_balanced'] = best_balanced[0]
        
        return recommendations
    
    def _create_evaluation_plots(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Create comprehensive evaluation visualizations
        
        Args:
            X_test, y_test: Test data
        """
        output_dir = Path(self.config['data']['output_dir']) / 'figures'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model comparison plot
        self._plot_model_comparison(output_dir)
        
        # ROC curves
        self._plot_roc_curves(output_dir)
        
        # Confusion matrices
        self._plot_confusion_matrices(output_dir)
    
    def _plot_model_comparison(self, output_dir: Path) -> None:
        """
        Create model comparison visualization
        
        Args:
            output_dir: Output directory for plots
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.evaluation_results.keys())
        
        # Prepare data for plotting
        comparison_data = []
        for metric in metrics:
            for model_name in model_names:
                if metric in self.evaluation_results[model_name]:
                    value = self.evaluation_results[model_name][metric]
                    if value is not None:
                        comparison_data.append({
                            'Model': model_name,
                            'Metric': metric.title(),
                            'Score': value
                        })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=df_comparison, x='Model', y='Score', hue='Metric')
            plt.title('Model Performance Comparison')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_roc_curves(self, output_dir: Path) -> None:
        """
        Plot ROC curves for all models that support probability prediction
        
        Args:
            output_dir: Output directory for plots
        """
        plt.figure(figsize=(10, 8))
        
        has_roc_data = False
        for model_name, results in self.evaluation_results.items():
            if 'roc_curve' in results and results['roc_curve']:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                roc_auc = results.get('roc_auc', 0)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                has_roc_data = True
        
        if has_roc_data:
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_confusion_matrices(self, output_dir: Path) -> None:
        """
        Plot confusion matrices for all models
        
        Args:
            output_dir: Output directory for plots
        """
        n_models = len(self.evaluation_results)
        if n_models == 0:
            return
            
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if i < len(axes) and 'confusion_matrix' in results:
                cm = np.array(results['confusion_matrix'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{model_name}\nConfusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.evaluation_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_results(self) -> None:
        """
        Save evaluation results to file
        """
        results_dir = Path(self.config['data']['output_dir']) / 'reports'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for programmatic access
        import json
        with open(results_dir / 'model_evaluation_results.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # Save human-readable report
        report_path = results_dir / 'model_evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in self.evaluation_results.items():
                f.write(f"{model_name.upper()}\n")
                f.write("-" * len(model_name) + "\n")
                f.write(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"Precision: {results.get('precision', 'N/A'):.4f}\n")
                f.write(f"Recall: {results.get('recall', 'N/A'):.4f}\n")
                f.write(f"F1 Score: {results.get('f1_score', 'N/A'):.4f}\n")
                f.write(f"ROC AUC: {results.get('roc_auc', 'N/A')}\n")
                f.write(f"CV Mean: {results.get('cross_val_mean', 'N/A')}\n")
                f.write(f"CV Std: {results.get('cross_val_std', 'N/A')}\n")
                f.write("\n")
        
        self.logger.info(f"Evaluation results saved to {results_dir}")
