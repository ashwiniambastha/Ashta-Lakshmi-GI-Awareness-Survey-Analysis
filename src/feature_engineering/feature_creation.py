"""
Feature engineering module for Ashta Lakshmi GI Survey
Creates new features and transforms existing ones for machine learning
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import KNNImputer

class FeatureEngineer:
    """
    Class for creating and transforming features for machine learning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FeatureEngineer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")
        
        feature_df = df.copy()
        
        # Create derived features
        feature_df = self._create_derived_features(feature_df)
        
        # Create categorical encodings
        feature_df = self._create_categorical_encodings(feature_df)
        
        # Create polynomial features
        feature_df = self._create_polynomial_features(feature_df)
        
        # Create interaction features
        feature_df = self._create_interaction_features(feature_df)
        
        # Create binned features
        feature_df = self._create_binned_features(feature_df)
        
        # Create aggregated features
        feature_df = self._create_aggregated_features(feature_df)
        
        # Scale numerical features
        feature_df = self._scale_features(feature_df)
        
        self.logger.info(f"Feature engineering completed. Features created: {feature_df.shape[1]}")
        return feature_df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing columns
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with derived features
        """
        feature_df = df.copy()
        
        # Age-related features
        feature_df['Age_Group'] = pd.cut(df['Age'], 
                                        bins=[0, 25, 35, 45, 55, 100], 
                                        labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elder'])
        
        # Experience-related features
        feature_df['Experience_Group'] = pd.cut(df['Years_of_Experience'], 
                                               bins=[0, 5, 15, 25, 50], 
                                               labels=['Novice', 'Intermediate', 'Experienced', 'Expert'])
        
        # Age-Experience ratio
        feature_df['Age_Experience_Ratio'] = df['Age'] / (df['Years_of_Experience'] + 1)
        feature_df['Experience_per_Age'] = df['Years_of_Experience'] / df['Age']
        
        # Career start age
        feature_df['Career_Start_Age'] = df['Age'] - df['Years_of_Experience']
        
        # Experience intensity (higher values = started career later in life)
        feature_df['Late_Career_Start'] = (feature_df['Career_Start_Age'] > 30).astype(int)
        
        # Digital adoption score (combination of e-commerce usage and other factors)
        feature_df['Digital_Adoption_Score'] = (
            df['Uses_Ecommerce_Binary'] * 2 + 
            (df['Age'] < 40).astype(int) * 1
        )
        
        # Awareness-Support correlation
        feature_df['Awareness_Support_Score'] = (
            df['GI_Aware_Binary'] * 2 + 
            df['Received_Subsidy_Binary'] * 1
        )
        
        # Overall engagement score
        feature_df['Engagement_Score'] = (
            df['GI_Aware_Binary'] + 
            df['Uses_Ecommerce_Binary'] + 
            df['Received_Subsidy_Binary']
        )
        
        self.logger.info("Derived features created")
        return feature_df
    
    def _create_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create various encodings for categorical variables
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with encoded categorical features
        """
        feature_df = df.copy()
        
        # One-hot encoding for State
        state_dummies = pd.get_dummies(df['State'], prefix='State')
        feature_df = pd.concat([feature_df, state_dummies], axis=1)
        
        # Label encoding for State (for tree-based models)
        le_state = LabelEncoder()
        feature_df['State_Encoded'] = le_state.fit_transform(df['State'])
        self.encoders['state_encoder'] = le_state
        
        # Target encoding for State (mean GI awareness by state)
        if 'GI_Aware_Binary' in df.columns:
            state_gi_mean = df.groupby('State')['GI_Aware_Binary'].mean()
            feature_df['State_GI_Target_Encoded'] = df['State'].map(state_gi_mean)
        
        # Frequency encoding for State
        state_counts = df['State'].value_counts()
        feature_df['State_Frequency'] = df['State'].map(state_counts)
        
        # Age group encoding
        if 'Age_Group' in feature_df.columns:
            age_group_dummies = pd.get_dummies(feature_df['Age_Group'], prefix='Age_Group')
            feature_df = pd.concat([feature_df, age_group_dummies], axis=1)
        
        # Experience group encoding
        if 'Experience_Group' in feature_df.columns:
            exp_group_dummies = pd.get_dummies(feature_df['Experience_Group'], prefix='Exp_Group')
            feature_df = pd.concat([feature_df, exp_group_dummies], axis=1)
        
        self.logger.info("Categorical encodings created")
        return feature_df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features for numerical variables
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with polynomial features
        """
        feature_df = df.copy()
        
        # Select numerical columns for polynomial features
        numerical_cols = ['Age', 'Years_of_Experience', 'Age_Experience_Ratio', 'Experience_per_Age']
        available_cols = [col for col in numerical_cols if col in df.columns]
        
        if available_cols:
            poly_degree = self.config.get('features', {}).get('polynomial_degree', 2)
            
            # Create polynomial features
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(df[available_cols])
            poly_feature_names = poly.get_feature_names_out(available_cols)
            
            # Add polynomial features to dataframe (excluding original features)
            original_feature_count = len(available_cols)
            new_poly_features = poly_features[:, original_feature_count:]
            new_poly_names = poly_feature_names[original_feature_count:]
            
            poly_df = pd.DataFrame(new_poly_features, columns=new_poly_names, index=df.index)
            feature_df = pd.concat([feature_df, poly_df], axis=1)
            
            self.logger.info(f"Created {len(new_poly_names)} polynomial features")
        
        return feature_df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with interaction features
        """
        feature_df = df.copy()
        
        # Age-Gender interaction
        feature_df['Age_Male_Interaction'] = df['Age'] * df['Gender_Binary']
        
        # Experience-Gender interaction
        feature_df['Experience_Male_Interaction'] = df['Years_of_Experience'] * df['Gender_Binary']
        
        # GI Awareness interactions
        if 'GI_Aware_Binary' in df.columns:
            feature_df['GI_Age_Interaction'] = df['GI_Aware_Binary'] * df['Age']
            feature_df['GI_Experience_Interaction'] = df['GI_Aware_Binary'] * df['Years_of_Experience']
            feature_df['GI_Gender_Interaction'] = df['GI_Aware_Binary'] * df['Gender_Binary']
        
        # Subsidy interactions
        if 'Received_Subsidy_Binary' in df.columns:
            feature_df['Subsidy_Age_Interaction'] = df['Received_Subsidy_Binary'] * df['Age']
            feature_df['Subsidy_Experience_Interaction'] = df['Received_Subsidy_Binary'] * df['Years_of_Experience']
        
        # E-commerce interactions
        if 'Uses_Ecommerce_Binary' in df.columns:
            feature_df['Ecommerce_Age_Interaction'] = df['Uses_Ecommerce_Binary'] * df['Age']
            feature_df['Ecommerce_Gender_Interaction'] = df['Uses_Ecommerce_Binary'] * df['Gender_Binary']
        
        self.logger.info("Interaction features created")
        return feature_df
    
    def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned versions of continuous features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with binned features
        """
        feature_df = df.copy()
        
        # Age bins
        feature_df['Age_Bin_5'] = pd.cut(df['Age'], bins=5, labels=False)
        feature_df['Age_Bin_10'] = pd.cut(df['Age'], bins=10, labels=False)
        
        # Experience bins
        feature_df['Experience_Bin_5'] = pd.cut(df['Years_of_Experience'], bins=5, labels=False)
        feature_df['Experience_Bin_10'] = pd.cut(df['Years_of_Experience'], bins=10, labels=False)
        
        # Quantile-based binning
        feature_df['Age_Quartiles'] = pd.qcut(df['Age'], q=4, labels=False)
        feature_df['Experience_Quartiles'] = pd.qcut(df['Years_of_Experience'], q=4, labels=False)
        
        # Custom bins based on domain knowledge
        feature_df['Young_Artisan'] = (df['Age'] <= 30).astype(int)
        feature_df['Senior_Artisan'] = (df['Age'] >= 50).astype(int)
        feature_df['Novice_Artisan'] = (df['Years_of_Experience'] <= 5).astype(int)
        feature_df['Expert_Artisan'] = (df['Years_of_Experience'] >= 20).astype(int)
        
        self.logger.info("Binned features created")
        return feature_df
    
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated features based on groupings
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with aggregated features
        """
        feature_df = df.copy()
        
        # Convert categorical columns back to their original types for calculations
        if 'State' in feature_df.columns and feature_df['State'].dtype.name == 'category':
            feature_df['State'] = feature_df['State'].astype(str)
        
        # Ensure numerical columns are numeric
        for col in ['Age', 'Years_of_Experience']:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        
        # State-level aggregations
        state_age_mean = feature_df.groupby('State')['Age'].mean()
        state_exp_mean = feature_df.groupby('State')['Years_of_Experience'].mean()
        
        feature_df['State_Avg_Age'] = feature_df['State'].map(state_age_mean)
        feature_df['State_Avg_Experience'] = feature_df['State'].map(state_exp_mean)
        
        # Deviation from state averages
        feature_df['Age_Deviation_from_State'] = feature_df['Age'] - feature_df['State_Avg_Age']
        feature_df['Experience_Deviation_from_State'] = feature_df['Years_of_Experience'] - feature_df['State_Avg_Experience']
        
        # Gender-level aggregations
        gender_gi_rate = df.groupby('Gender')['GI_Aware_Binary'].mean()
        gender_ecom_rate = df.groupby('Gender')['Uses_Ecommerce_Binary'].mean()
        
        feature_df['Gender_GI_Rate'] = df['Gender'].map(gender_gi_rate)
        feature_df['Gender_Ecommerce_Rate'] = df['Gender'].map(gender_ecom_rate)
        
        # Age group aggregations
        if 'Age_Group' in feature_df.columns:
            age_group_gi = df.groupby('Age_Group')['GI_Aware_Binary'].mean()
            feature_df['Age_Group_GI_Rate'] = feature_df['Age_Group'].map(age_group_gi)
        
        self.logger.info("Aggregated features created")
        return feature_df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with scaled features
        """
        feature_df = df.copy()
        
        # Identify numerical columns to scale
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude binary and categorical encoded columns from scaling
        exclude_cols = [col for col in numerical_cols if 
                       'Binary' in col or 'Encoded' in col or 
                       col.startswith('State_') and col != 'State_Encoded' or
                       col.endswith('_Bin_5') or col.endswith('_Bin_10') or
                       col.endswith('_Quartiles')]
        
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_df[cols_to_scale])
            
            # Create scaled feature names
            scaled_feature_names = [f"{col}_scaled" for col in cols_to_scale]
            
            # Add scaled features to dataframe
            scaled_df = pd.DataFrame(scaled_features, columns=scaled_feature_names, index=feature_df.index)
            feature_df = pd.concat([feature_df, scaled_df], axis=1)
            
            # Store scaler for future use
            self.scalers['standard_scaler'] = scaler
            self.scalers['scaled_columns'] = cols_to_scale
            
            self.logger.info(f"Scaled {len(cols_to_scale)} numerical features")
        
        return feature_df
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed dataframe
        """
        feature_df = self.create_features(df)
        
        # Apply saved scalers
        if 'standard_scaler' in self.scalers and 'scaled_columns' in self.scalers:
            scaler = self.scalers['standard_scaler']
            cols_to_scale = self.scalers['scaled_columns']
            
            # Only scale columns that exist in the new data
            available_cols = [col for col in cols_to_scale if col in feature_df.columns]
            
            if available_cols:
                scaled_features = scaler.transform(feature_df[available_cols])
                scaled_feature_names = [f"{col}_scaled" for col in available_cols]
                
                scaled_df = pd.DataFrame(scaled_features, columns=scaled_feature_names, index=feature_df.index)
                feature_df = pd.concat([feature_df, scaled_df], axis=1)
        
        return feature_df
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about created features
        
        Returns:
            Dictionary containing feature information
        """
        return {
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'feature_types': {
                'derived': ['Age_Group', 'Experience_Group', 'Age_Experience_Ratio', 'Career_Start_Age'],
                'polynomial': 'Created based on configuration',
                'interaction': ['Age_Male_Interaction', 'GI_Age_Interaction', 'Subsidy_Age_Interaction'],
                'binned': ['Age_Bin_5', 'Experience_Quartiles', 'Young_Artisan', 'Expert_Artisan'],
                'aggregated': ['State_Avg_Age', 'Age_Deviation_from_State', 'Gender_GI_Rate']
            }
        }
