"""
EDA visualization module for Ashta Lakshmi GI Survey
Creates comprehensive exploratory data analysis plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class EDAVisualizer:
    """
    Class for creating comprehensive EDA visualizations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EDAVisualizer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(config['data']['output_dir']) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette(config.get('plots', {}).get('color_palette', 'Set2'))
        self.figsize = config.get('plots', {}).get('figsize', [10, 8])
        
    def create_all_plots(self, df: pd.DataFrame, stat_results: Dict[str, Any], 
                        corr_results: Dict[str, Any]) -> None:
        """
        Create all EDA visualizations
        
        Args:
            df: Cleaned dataframe
            stat_results: Statistical analysis results
            corr_results: Correlation analysis results
        """
        self.logger.info("Creating EDA visualizations...")
        
        # Basic distribution plots
        self._create_distribution_plots(df)
        
        # Demographic analysis plots
        self._create_demographic_plots(df)
        
        # State-wise analysis plots
        self._create_state_analysis_plots(df)
        
        # Correlation heatmaps
        self._create_correlation_heatmaps(corr_results)
        
        # Statistical test visualizations
        self._create_statistical_plots(df, stat_results)
        
        # Interactive plots
        self._create_interactive_plots(df)
        
        # Summary dashboard
        self._create_summary_dashboard(df, stat_results)
        
        self.logger.info("EDA visualizations completed")
    
    def _create_distribution_plots(self, df: pd.DataFrame) -> None:
        """
        Create distribution plots for numerical variables
        
        Args:
            df: Input dataframe
        """
        # Age and Experience distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution
        sns.histplot(data=df, x='Age', kde=True, ax=axes[0,0])
        axes[0,0].set_title('Distribution of Age')
        axes[0,0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
        axes[0,0].legend()
        
        # Years of Experience distribution
        sns.histplot(data=df, x='Years_of_Experience', kde=True, ax=axes[0,1])
        axes[0,1].set_title('Distribution of Years of Experience')
        axes[0,1].axvline(df['Years_of_Experience'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["Years_of_Experience"].mean():.1f}')
        axes[0,1].legend()
        
        # Age vs Experience scatter
        sns.scatterplot(data=df, x='Age', y='Years_of_Experience', hue='Gender', ax=axes[1,0])
        axes[1,0].set_title('Age vs Years of Experience')
        
        # Box plots
        df_melt = pd.melt(df, value_vars=['Age', 'Years_of_Experience'], var_name='Variable', value_name='Value')
        sns.boxplot(data=df_melt, x='Variable', y='Value', ax=axes[1,1])
        axes[1,1].set_title('Box Plots of Numerical Variables')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distribution_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_demographic_plots(self, df: pd.DataFrame) -> None:
        """
        Create demographic analysis plots
        
        Args:
            df: Input dataframe
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Gender distribution
        gender_counts = df['Gender'].value_counts()
        axes[0,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Gender Distribution')
        
        # GI Awareness by Gender
        gi_gender = pd.crosstab(df['Gender'], df['GI_Aware'], normalize='index') * 100
        gi_gender.plot(kind='bar', ax=axes[0,1], width=0.7)
        axes[0,1].set_title('GI Awareness by Gender (%)')
        axes[0,1].set_ylabel('Percentage')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='GI Aware')
        
        # E-commerce usage by Gender
        ecom_gender = pd.crosstab(df['Gender'], df['Uses_Ecommerce'], normalize='index') * 100
        ecom_gender.plot(kind='bar', ax=axes[0,2], width=0.7)
        axes[0,2].set_title('E-commerce Usage by Gender (%)')
        axes[0,2].set_ylabel('Percentage')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].legend(title='Uses E-commerce')
        
        # Age distribution by Gender
        sns.violinplot(data=df, x='Gender', y='Age', ax=axes[1,0])
        axes[1,0].set_title('Age Distribution by Gender')
        
        # Experience distribution by Gender
        sns.violinplot(data=df, x='Gender', y='Years_of_Experience', ax=axes[1,1])
        axes[1,1].set_title('Experience Distribution by Gender')
        
        # Subsidy distribution by Gender
        subsidy_gender = pd.crosstab(df['Gender'], df['Received_Subsidy'], normalize='index') * 100
        subsidy_gender.plot(kind='bar', ax=axes[1,2], width=0.7)
        axes[1,2].set_title('Subsidy Receipt by Gender (%)')
        axes[1,2].set_ylabel('Percentage')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].legend(title='Received Subsidy')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'demographic_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_state_analysis_plots(self, df: pd.DataFrame) -> None:
        """
        Create state-wise analysis plots
        
        Args:
            df: Input dataframe
        """
        # Calculate state-wise statistics
        state_stats = df.groupby('State').agg({
            'GI_Aware': lambda x: (x == 'Yes').mean() * 100,
            'Uses_Ecommerce': lambda x: (x == 'Yes').mean() * 100,
            'Received_Subsidy': lambda x: (x == 'Yes').mean() * 100,
            'Age': 'mean',
            'Years_of_Experience': 'mean'
        }).round(2)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # GI Awareness by State
        state_stats['GI_Aware'].plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('GI Awareness by State (%)')
        axes[0,0].set_ylabel('Percentage')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # E-commerce usage by State
        state_stats['Uses_Ecommerce'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('E-commerce Usage by State (%)')
        axes[0,1].set_ylabel('Percentage')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Subsidy receipt by State
        state_stats['Received_Subsidy'].plot(kind='bar', ax=axes[0,2], color='lightcoral')
        axes[0,2].set_title('Subsidy Receipt by State (%)')
        axes[0,2].set_ylabel('Percentage')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # Average Age by State
        state_stats['Age'].plot(kind='bar', ax=axes[1,0], color='gold')
        axes[1,0].set_title('Average Age by State')
        axes[1,0].set_ylabel('Years')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Average Experience by State
        state_stats['Years_of_Experience'].plot(kind='bar', ax=axes[1,1], color='plum')
        axes[1,1].set_title('Average Experience by State')
        axes[1,1].set_ylabel('Years')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # State distribution
        state_counts = df['State'].value_counts()
        axes[1,2].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
        axes[1,2].set_title('Distribution of Artisans by State')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'state_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_heatmaps(self, corr_results: Dict[str, Any]) -> None:
        """
        Create correlation heatmaps
        
        Args:
            corr_results: Correlation analysis results
        """
        # Pearson correlation heatmap
        pearson_corr = pd.DataFrame(corr_results['pearson_matrix']['correlation_matrix'])
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Pearson correlation
        sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[0], fmt='.3f')
        axes[0].set_title('Pearson Correlation Matrix')
        
        # Spearman correlation
        spearman_corr = pd.DataFrame(corr_results['spearman_matrix']['correlation_matrix'])
        sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=axes[1], fmt='.3f')
        axes[1].set_title('Spearman Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_plots(self, df: pd.DataFrame, stat_results: Dict[str, Any]) -> None:
        """
        Create plots for statistical test results
        
        Args:
            df: Input dataframe
            stat_results: Statistical analysis results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Chi-square test visualization (Gender vs GI Awareness)
        gender_gi_crosstab = pd.crosstab(df['Gender'], df['GI_Aware'])
        gender_gi_crosstab.plot(kind='bar', ax=axes[0,0], width=0.7)
        axes[0,0].set_title('Gender vs GI Awareness (Chi-square test)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='GI Aware')
        
        # Confidence intervals plot
        if 'confidence_intervals' in stat_results:
            ci_data = []
            labels = []
            for key, ci_info in stat_results['confidence_intervals'].items():
                if 'proportion' in key:
                    ci_data.append([ci_info['confidence_interval'][0], 
                                   ci_info['estimate'], 
                                   ci_info['confidence_interval'][1]])
                    labels.append(key.replace('_Binary_proportion', ''))
            
            if ci_data:
                ci_df = pd.DataFrame(ci_data, columns=['Lower', 'Estimate', 'Upper'], index=labels)
                for i, (idx, row) in enumerate(ci_df.iterrows()):
                    axes[0,1].errorbar(i, row['Estimate'], 
                                     yerr=[[row['Estimate'] - row['Lower']], 
                                           [row['Upper'] - row['Estimate']]], 
                                     fmt='o', capsize=5, capthick=2)
                axes[0,1].set_xticks(range(len(labels)))
                axes[0,1].set_xticklabels(labels, rotation=45)
                axes[0,1].set_title('95% Confidence Intervals for Proportions')
                axes[0,1].set_ylabel('Proportion')
        
        # Age distribution by GI Awareness
        sns.boxplot(data=df, x='GI_Aware', y='Age', ax=axes[1,0])
        axes[1,0].set_title('Age Distribution by GI Awareness')
        
        # Experience distribution by GI Awareness
        sns.boxplot(data=df, x='GI_Aware', y='Years_of_Experience', ax=axes[1,1])
        axes[1,1].set_title('Experience Distribution by GI Awareness')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_interactive_plots(self, df: pd.DataFrame) -> None:
        """
        Create interactive plots using Plotly
        
        Args:
            df: Input dataframe
        """
        # Interactive scatter plot: Age vs Experience colored by State
        fig = px.scatter(df, x='Age', y='Years_of_Experience', color='State', 
                        size='GI_Aware_Binary', hover_data=['Gender', 'Uses_Ecommerce'],
                        title='Age vs Experience by State (Size = GI Awareness)')
        fig.write_html(str(self.output_dir / 'interactive_scatter.html'))
        
        # Interactive state-wise analysis
        state_summary = df.groupby('State').agg({
            'GI_Aware_Binary': 'mean',
            'Uses_Ecommerce_Binary': 'mean',
            'Received_Subsidy_Binary': 'mean',
            'Age': 'mean',
            'Artisan_ID': 'count'
        }).round(3)
        state_summary.columns = ['GI_Awareness_Rate', 'Ecommerce_Rate', 'Subsidy_Rate', 'Avg_Age', 'Count']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('GI Awareness Rate', 'E-commerce Usage Rate', 
                           'Subsidy Receipt Rate', 'Average Age'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar plots
        fig.add_trace(go.Bar(x=state_summary.index, y=state_summary['GI_Awareness_Rate'], name='GI Awareness'), 
                     row=1, col=1)
        fig.add_trace(go.Bar(x=state_summary.index, y=state_summary['Ecommerce_Rate'], name='E-commerce'), 
                     row=1, col=2)
        fig.add_trace(go.Bar(x=state_summary.index, y=state_summary['Subsidy_Rate'], name='Subsidy'), 
                     row=2, col=1)
        fig.add_trace(go.Bar(x=state_summary.index, y=state_summary['Avg_Age'], name='Age'), 
                     row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="State-wise Analysis Dashboard")
        fig.write_html(str(self.output_dir / 'interactive_state_analysis.html'))
    
    def _create_summary_dashboard(self, df: pd.DataFrame, stat_results: Dict[str, Any]) -> None:
        """
        Create summary dashboard with key metrics
        
        Args:
            df: Input dataframe
            stat_results: Statistical analysis results
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Key metrics
        total_artisans = len(df)
        gi_awareness_rate = (df['GI_Aware'] == 'Yes').mean() * 100
        ecommerce_rate = (df['Uses_Ecommerce'] == 'Yes').mean() * 100
        subsidy_rate = (df['Received_Subsidy'] == 'Yes').mean() * 100
        avg_age = df['Age'].mean()
        avg_experience = df['Years_of_Experience'].mean()
        gender_gap = abs(df[df['Gender'] == 'Male']['GI_Aware_Binary'].mean() - 
                        df[df['Gender'] == 'Female']['GI_Aware_Binary'].mean()) * 100
        
        # Metric cards
        metrics = [
            ('Total Artisans', total_artisans, 'lightblue'),
            ('GI Awareness %', gi_awareness_rate, 'lightgreen'),
            ('E-commerce Usage %', ecommerce_rate, 'lightcoral'),
            ('Subsidy Recipients %', subsidy_rate, 'lightyellow'),
            ('Average Age', avg_age, 'lightpink'),
            ('Average Experience', avg_experience, 'lightgray'),
            ('Gender Awareness Gap %', gender_gap, 'orange'),
            ('States Covered', df['State'].nunique(), 'lightcyan')
        ]
        
        for i, (metric, value, color) in enumerate(metrics):
            row, col = divmod(i, 4)
            axes[row, col].text(0.5, 0.5, f'{metric}\n{value:.1f}', 
                               ha='center', va='center', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)
            axes[row, col].axis('off')
        
        plt.suptitle('Ashta Lakshmi GI Survey - Key Metrics Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
