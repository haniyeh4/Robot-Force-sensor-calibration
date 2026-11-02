#!/usr/bin/env python3
"""
MLP Model for Dynamic Force Prediction
=====================================

This script trains an MLP (Multi-Layer Perceptron) model to predict dynamic forces
in a KUKA robotic arm system using 9 input features:
- POS[3:6]: Orientation angles (rx, ry, rz)
- X_DoubleDot_Cmds[0:6]: 6DOF acceleration commands

Output: FF_EE_Dynamic[0:6] (6DOF dynamic forces)

Author: Generated for KUKA Data Processing Paper
Date: 2024
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# ==========================
# Configuration (Edit here)
# ==========================
# Base data directory containing JSONL files
EXP_NAME = 'Aug29'
DATA_DIR = "data/" + EXP_NAME

# Output directory for models, plots, and reports
OUTPUT_DIR = "models/" + EXP_NAME + "/mlp"

# Explicit dataset lists (filenames within DATA_DIR)
# Edit these lists to control which files are used for training and testing.
TEST_DATASETS = [
    '2_DyF_L1.jsonl'
]

TRAIN_DATASETS = [
    '2_DyF_L2.jsonl', '2_DyF_L3.jsonl', '2_DyF_L4.jsonl', '2_DyF_L5.jsonl',
    '2_DyF_L6.jsonl', '2_DyF_L7.jsonl', '2_DyF_L8.jsonl', '2_DyF_L9.jsonl',
    '2_DyF_L10.jsonl', '2_DyF_L11.jsonl'
]

class MLPDynamicForcePredictor:
    """
    MLP-based dynamic force prediction model for KUKA robotic arm.
    """
    
    def __init__(self, data_dir=DATA_DIR, output_dir=OUTPUT_DIR, train_datasets=TRAIN_DATASETS, test_datasets=TEST_DATASETS):
        """
        Initialize the MLP predictor.
        
        Args:
            data_dir (str): Directory containing the JSONL data files
            output_dir (str): Directory to save results and models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature and target column names
        self.feature_cols = [
            'POS_3', 'POS_4', 'POS_5',  # rx, ry, rz
            'X_DoubleDot_Cmds_0', 'X_DoubleDot_Cmds_1', 'X_DoubleDot_Cmds_2',
            'X_DoubleDot_Cmds_3', 'X_DoubleDot_Cmds_4', 'X_DoubleDot_Cmds_5'
        ]
        self.target_cols = [
            'FF_EE_Dynamic_0', 'FF_EE_Dynamic_1', 'FF_EE_Dynamic_2',
            'FF_EE_Dynamic_3', 'FF_EE_Dynamic_4', 'FF_EE_Dynamic_5'
        ]
        
    def load_data(self, datasets=None, split_method='dataset_based'):
        """
        Load and combine data from multiple JSONL files.
        
        Args:
            datasets (list): List of dataset names to load. If None, loads all available.
            split_method (str): 'dataset_based' for L1 test/L2-L11 train, 'all' for all datasets
            
        Returns:
            tuple: (train_df, test_df) or (df, None) depending on split_method
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)
        
        if datasets is None:
            # If explicit lists are provided via config, use them; otherwise auto-discover
            if split_method == 'dataset_based' and (self.train_datasets or self.test_datasets):
                # We'll construct from provided lists below
                datasets = None
            else:
                datasets = [f for f in os.listdir(self.data_dir) if f.startswith('2_DyF_L')]
                datasets.sort()
        
        print(f"Available datasets: {datasets}")
        
        if split_method == 'dataset_based':
            # Build train/test lists honoring top-level config if provided
            if self.test_datasets:
                test_datasets = list(self.test_datasets)
            else:
                test_datasets = ['2_DyF_L1.jsonl']

            if self.train_datasets:
                train_datasets = list(self.train_datasets)
            else:
                if datasets is None:
                    # Auto-discover then exclude tests
                    discovered = [f for f in os.listdir(self.data_dir) if f.startswith('2_DyF_L')]
                    discovered.sort()
                    train_datasets = [d for d in discovered if d not in set(test_datasets)]
                else:
                    train_datasets = [d for d in datasets if d not in set(test_datasets)]
            
            print(f"Test dataset: {test_datasets}")
            print(f"Training datasets: {train_datasets}")
            
            # Load test data
            test_data = []
            test_records = 0
            for dataset in test_datasets:
                file_path = os.path.join(self.data_dir, dataset)
                if not os.path.exists(file_path):
                    print(f"Warning: {file_path} not found, skipping...")
                    continue
                    
                print(f"Loading test data from {dataset}...")
                with open(file_path, 'r') as f:
                    for line in tqdm(f, desc=f"Loading {dataset}"):
                        try:
                            record = json.loads(line.strip())
                            test_data.append(record)
                            test_records += 1
                        except json.JSONDecodeError:
                            continue
            
            # Load training data
            train_data = []
            train_records = 0
            for dataset in train_datasets:
                file_path = os.path.join(self.data_dir, dataset)
                if not os.path.exists(file_path):
                    print(f"Warning: {file_path} not found, skipping...")
                    continue
                    
                print(f"Loading training data from {dataset}...")
                with open(file_path, 'r') as f:
                    for line in tqdm(f, desc=f"Loading {dataset}"):
                        try:
                            record = json.loads(line.strip())
                            train_data.append(record)
                            train_records += 1
                        except json.JSONDecodeError:
                            continue
            
            print(f"\nTest records loaded: {test_records:,}")
            print(f"Training records loaded: {train_records:,}")
            print(f"Total records: {test_records + train_records:,}")
            
            # Convert to DataFrames
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            
            print(f"\nTraining DataFrame shape: {train_df.shape}")
            print(f"Test DataFrame shape: {test_df.shape}")
            
            # Display basic statistics
            print(f"\nTraining Data Overview:")
            print(f"- Time range: {train_df['timestamp'].min():.2f} to {train_df['timestamp'].max():.2f}")
            print(f"- Duration: {(train_df['timestamp'].max() - train_df['timestamp'].min())/60:.1f} minutes")
            
            print(f"\nTest Data Overview:")
            print(f"- Time range: {test_df['timestamp'].min():.2f} to {test_df['timestamp'].max():.2f}")
            print(f"- Duration: {(test_df['timestamp'].max() - test_df['timestamp'].min())/60:.1f} minutes")
            
            return train_df, test_df
            
        else:
            # Load all data (original behavior)
            print(f"Loading all datasets: {datasets}")
            
            all_data = []
            total_records = 0
            
            for dataset in datasets:
                file_path = os.path.join(self.data_dir, dataset)
                if not os.path.exists(file_path):
                    print(f"Warning: {file_path} not found, skipping...")
                    continue
                    
                print(f"Loading {dataset}...")
                dataset_records = 0
                
                with open(file_path, 'r') as f:
                    for line in tqdm(f, desc=f"Loading {dataset}"):
                        try:
                            record = json.loads(line.strip())
                            all_data.append(record)
                            dataset_records += 1
                        except json.JSONDecodeError:
                            continue
                
                print(f"  - {dataset_records:,} records loaded")
                total_records += dataset_records
            
            print(f"\nTotal records loaded: {total_records:,}")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            print(f"DataFrame shape: {df.shape}")
            
            # Display basic statistics
            print(f"\nData Overview:")
            print(f"- Time range: {df['timestamp'].min():.2f} to {df['timestamp'].max():.2f}")
            print(f"- Duration: {(df['timestamp'].max() - df['timestamp'].min())/60:.1f} minutes")
            
            return df, None
    
    def preprocess_data(self, train_df, test_df=None):
        """
        Preprocess the loaded data for MLP training.
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data (optional)
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test) or (X, y, None, None)
        """
        print("\n" + "=" * 60)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 60)
        
        def extract_features_targets(df, data_type=""):
            print(f"Processing {data_type} data...")
            
            # Extract features: orientation + acceleration commands
            X_data = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {data_type} features"):
                features = (
                    row['POS'][3:6] +  # rx, ry, rz
                    row['X_DoubleDot_Cmds']  # 6DOF accelerations
                )
                X_data.append(features)
            
            X = np.array(X_data)
            
            # Extract targets: dynamic forces
            y_data = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {data_type} targets"):
                targets = row['FF_EE_Dynamic']
                y_data.append(targets)
            
            y = np.array(y_data)
            
            print(f"{data_type.capitalize()} feature matrix shape: {X.shape}")
            print(f"{data_type.capitalize()} target matrix shape: {y.shape}")
            
            return X, y
        
        # Process training data
        X_train, y_train = extract_features_targets(train_df, "training")
        
        # Process test data if provided
        if test_df is not None:
            X_test, y_test = extract_features_targets(test_df, "test")
        else:
            X_test, y_test = None, None
        
        # Feature statistics
        print(f"\nTraining Feature Statistics:")
        feature_names = ['rx', 'ry', 'rz', 'ax', 'ay', 'az', 'αx', 'αy', 'αz']
        for i, name in enumerate(feature_names):
            print(f"  {name:3s}: mean={X_train[:, i].mean():8.4f}, std={X_train[:, i].std():8.4f}, "
                  f"min={X_train[:, i].min():8.4f}, max={X_train[:, i].max():8.4f}")
        
        # Target statistics
        print(f"\nTraining Target Statistics:")
        target_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        for i, name in enumerate(target_names):
            print(f"  {name:2s}: mean={y_train[:, i].mean():8.4f}, std={y_train[:, i].std():8.4f}, "
                  f"min={y_train[:, i].min():8.4f}, max={y_train[:, i].max():8.4f}")
        
        if X_test is not None:
            print(f"\nTest Feature Statistics:")
            for i, name in enumerate(feature_names):
                print(f"  {name:3s}: mean={X_test[:, i].mean():8.4f}, std={X_test[:, i].std():8.4f}, "
                      f"min={X_test[:, i].min():8.4f}, max={X_test[:, i].max():8.4f}")
            
            print(f"\nTest Target Statistics:")
            for i, name in enumerate(target_names):
                print(f"  {name:2s}: mean={y_test[:, i].mean():8.4f}, std={y_test[:, i].std():8.4f}, "
                      f"min={y_test[:, i].min():8.4f}, max={y_test[:, i].max():8.4f}")
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train, X_test=None, y_test=None, random_state=42):
        """
        Train the MLP model with hyperparameter optimization.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_test (np.array): Test features (optional)
            y_test (np.array): Test targets (optional)
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        print("\n" + "=" * 60)
        print("STEP 3: MODEL TRAINING")
        print("=" * 60)
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        if X_test is not None:
            print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = None
        
        # Constrain to exactly two hidden layers to ensure W1–W3 export shape
        param_grid = {
            'hidden_layer_sizes': [(100, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'learning_rate': ['adaptive']
        }
        
        print("\nPerforming hyperparameter optimization...")
        print(f"Testing {len(param_grid['hidden_layer_sizes']) * len(param_grid['activation']) * len(param_grid['alpha'])} combinations...")
        
        # Create MLP regressor
        mlp = MLPRegressor(random_state=random_state, max_iter=1000)
        
        # Grid search with cross-validation and progress bar
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0  # Reduced verbosity
        )
        
        # Fit with progress tracking
        print("Training models...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"Best CV score (MSE): {-grid_search.best_score_:.6f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model and generate comprehensive metrics.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation results
        """
        print("\n" + "=" * 60)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Overall Performance:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
        
        # Per-axis performance
        print(f"\nPer-Axis Performance:")
        target_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        
        axis_results = {}
        for i, name in enumerate(target_names):
            axis_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            axis_rmse = np.sqrt(axis_mse)
            axis_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            axis_r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            print(f"  {name:2s}: MSE={axis_mse:.6f}, RMSE={axis_rmse:.6f}, "
                  f"MAE={axis_mae:.6f}, R²={axis_r2:.6f}")
            
            axis_results[name] = {
                'mse': axis_mse,
                'rmse': axis_rmse,
                'mae': axis_mae,
                'r2': axis_r2
            }
        
        # Store results
        self.results = {
            'overall': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'per_axis': axis_results,
            'predictions': {
                'y_test': y_test,
                'y_pred': y_pred
            }
        }
        
        return self.results
    
    def create_visualizations(self, time_series_start=0, time_series_length=1000):
        """
        Create comprehensive visualizations for the paper.
        
        Args:
            time_series_start (int): Starting index for time series sample
            time_series_length (int): Number of points to show in time series
        """
        print("\n" + "=" * 60)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("=" * 60)
        
        y_test = self.results['predictions']['y_test']
        y_pred = self.results['predictions']['y_pred']
        target_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        
        print(f"Time series sample: starting at index {time_series_start}, length {time_series_length}")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Prediction vs Actual Scatter Plots
        print("Creating prediction vs actual scatter plots...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(target_names):
            ax = axes[i]
            ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=1)
            
            # Perfect prediction line
            min_val = min(y_test[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel(f'Actual {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} - R² = {self.results["per_axis"][name]["r2"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mlp_prediction_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Error Distribution Histograms
        print("Creating error distribution histograms...")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(target_names):
            ax = axes[i]
            errors = y_pred[:, i] - y_test[:, i]
            ax.hist(errors, bins=50, alpha=0.7, density=True)
            ax.axvline(0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel(f'Prediction Error ({name})')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Error Distribution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mlp_error_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Performance Comparison Bar Chart
        print("Creating performance comparison chart...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['mse', 'rmse', 'mae', 'r2']
        metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            values = [self.results['per_axis'][name][metric] for name in target_names]
            bars = ax.bar(target_names, values, alpha=0.7)
            
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} by Axis')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mlp_performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Time Series Sample (customizable section)
        print("Creating time series sample plot...")
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Calculate actual sample size and bounds
        end_idx = min(time_series_start + time_series_length, len(y_test))
        actual_length = end_idx - time_series_start
        x_axis = np.arange(actual_length)
        
        print(f"Showing samples {time_series_start} to {end_idx-1} ({actual_length} points)")
        
        for i, name in enumerate(target_names):
            ax = axes[i]
            ax.plot(x_axis, y_test[time_series_start:end_idx, i], 'b-', label='Actual', alpha=0.7, linewidth=1)
            ax.plot(x_axis, y_pred[time_series_start:end_idx, i], 'r--', label='Predicted', alpha=0.8, linewidth=1)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'{name}')
            ax.set_title(f'{name} - Time Series Sample (samples {time_series_start}-{end_idx-1})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mlp_time_series_sample.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to: {self.output_dir}")
    
    def create_time_series_plot(self, start_idx=0, length=1000, save_name=None):
        """
        Create a custom time series plot for a specific section of the data.
        
        Args:
            start_idx (int): Starting index for the time series
            length (int): Number of points to show
            save_name (str): Custom filename (optional)
        """
        if not hasattr(self, 'results') or 'predictions' not in self.results:
            print("Error: No results available. Run the complete pipeline first.")
            return
        
        y_test = self.results['predictions']['y_test']
        y_pred = self.results['predictions']['y_pred']
        target_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        
        print(f"Creating time series plot: samples {start_idx} to {start_idx + length - 1}")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Calculate actual bounds
        end_idx = min(start_idx + length, len(y_test))
        actual_length = end_idx - start_idx
        x_axis = np.arange(actual_length)
        
        for i, name in enumerate(target_names):
            ax = axes[i]
            ax.plot(x_axis, y_test[start_idx:end_idx, i], 'b-', label='Actual', alpha=0.7, linewidth=1)
            ax.plot(x_axis, y_pred[start_idx:end_idx, i], 'r--', label='Predicted', alpha=0.8, linewidth=1)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'{name}')
            ax.set_title(f'{name} - Time Series Sample (samples {start_idx}-{end_idx-1})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with custom name or default
        if save_name is None:
            save_name = f'mlp_time_series_sample_{start_idx}_{end_idx-1}.png'
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Time series plot saved to: {save_path}")
    
    def save_results(self):
        """
        Save model, scaler, and results to files.
        """
        print("\n" + "=" * 60)
        print("STEP 6: SAVING RESULTS")
        print("=" * 60)
        
        # Save model
        model_path = os.path.join(self.output_dir, 'mlp_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.output_dir, 'mlp_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save results as JSON
        results_path = os.path.join(self.output_dir, 'mlp_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self.results.copy()
        json_results['predictions'] = {
            'y_test': self.results['predictions']['y_test'].tolist(),
            'y_pred': self.results['predictions']['y_pred'].tolist()
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Save summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report for the paper.
        """
        report_path = os.path.join(self.output_dir, 'mlp_summary_report.txt')
        
        # Calculate additional statistics
        y_test = self.results['predictions']['y_test']
        y_pred = self.results['predictions']['y_pred']
        errors = y_pred - y_test
        
        # Overall error statistics
        abs_errors = np.abs(errors)
        max_error = np.max(abs_errors)
        error_95th = np.percentile(abs_errors, 95)
        error_99th = np.percentile(abs_errors, 99)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MLP DYNAMIC FORCE PREDICTION - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Architecture: Multi-Layer Perceptron (MLP)\n")
            f.write(f"Input Features: 9 (rx, ry, rz + 6DOF accelerations)\n")
            f.write(f"Output Targets: 6 (6DOF dynamic forces)\n")
            f.write(f"Hidden Layers: {self.model.hidden_layer_sizes}\n")
            f.write(f"Activation: {self.model.activation}\n")
            f.write(f"Solver: {self.model.solver}\n")
            f.write(f"Regularization (α): {self.model.alpha}\n")
            f.write(f"Training Samples: {len(y_test):,}\n\n")
            
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            overall = self.results['overall']
            f.write(f"MSE:  {overall['mse']:.6f}\n")
            f.write(f"RMSE: {overall['rmse']:.6f}\n")
            f.write(f"MAE:  {overall['mae']:.6f}\n")
            f.write(f"R²:   {overall['r2']:.6f}\n")
            f.write(f"Max Absolute Error: {max_error:.6f}\n")
            f.write(f"95th Percentile Error: {error_95th:.6f}\n")
            f.write(f"99th Percentile Error: {error_99th:.6f}\n\n")
            
            f.write("PER-AXIS PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            target_names = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
            for i, name in enumerate(target_names):
                axis = self.results['per_axis'][name]
                axis_errors = np.abs(errors[:, i])
                axis_max_error = np.max(axis_errors)
                axis_95th = np.percentile(axis_errors, 95)
                axis_99th = np.percentile(axis_errors, 99)
                
                f.write(f"{name:2s}: MSE={axis['mse']:.6f}, RMSE={axis['rmse']:.6f}, "
                       f"MAE={axis['mae']:.6f}, R²={axis['r2']:.6f}\n")
                f.write(f"    Max Error: {axis_max_error:.6f}, 95th%: {axis_95th:.6f}, 99th%: {axis_99th:.6f}\n")
            
            f.write(f"\nERROR DISTRIBUTION ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Mean Absolute Error: {overall['mae']:.6f}\n")
            f.write(f"Standard Deviation of Errors: {np.std(errors):.6f}\n")
            f.write(f"Error Range: [{np.min(errors):.6f}, {np.max(errors):.6f}]\n")
            f.write(f"Percentage of predictions within ±0.01: {np.mean(np.abs(errors) <= 0.01)*100:.1f}%\n")
            f.write(f"Percentage of predictions within ±0.05: {np.mean(np.abs(errors) <= 0.05)*100:.1f}%\n")
            f.write(f"Percentage of predictions within ±0.1: {np.mean(np.abs(errors) <= 0.1)*100:.1f}%\n\n")
            
            f.write("BASELINE MODEL:\n")
            f.write("-" * 15 + "\n")
            f.write("This is the baseline MLP model using only orientation and acceleration\n")
            f.write("features. Compare with MLP+Velocity model to assess velocity contribution.\n\n")
            
            f.write(f"FILES GENERATED:\n")
            f.write("-" * 15 + "\n")
            f.write("- mlp_model.pkl: Trained MLP model\n")
            f.write("- mlp_scaler.pkl: Feature scaler\n")
            f.write("- mlp_results.json: Detailed results\n")
            f.write("- mlp_summary_report.txt: This report\n")
            f.write("- mlp_prediction_scatter.png: Scatter plots\n")
            f.write("- mlp_error_distribution.png: Error histograms\n")
            f.write("- mlp_performance_comparison.png: Performance bars\n")
            f.write("- mlp_time_series_sample.png: Time series sample\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def run_complete_pipeline(self, datasets=None, split_method='dataset_based'):
        """
        Run the complete MLP training and evaluation pipeline.
        
        Args:
            datasets (list): List of datasets to use. If None, uses all available.
            split_method (str): 'dataset_based' for L1 test/L2-L11 train, 'all' for all datasets
        """
        print("MLP DYNAMIC FORCE PREDICTION PIPELINE")
        print("=" * 60)
        print("Starting complete training and evaluation pipeline...\n")
        
        try:
            # Step 1: Load data
            if split_method == 'dataset_based':
                train_df, test_df = self.load_data(datasets, split_method='dataset_based')
            else:
                df, _ = self.load_data(datasets, split_method='all')
                train_df, test_df = df, None
            
            # Step 2: Preprocess data
            if test_df is not None:
                X_train, y_train, X_test, y_test = self.preprocess_data(train_df, test_df)
            else:
                X, y = self.preprocess_data(train_df)
                X_train, y_train, X_test, y_test = X, y, None, None
            
            # Step 3: Train model
            X_train_scaled, X_test_scaled, y_train, y_test = self.train_model(
                X_train, y_train, X_test, y_test
            )
            
            # Step 4: Evaluate model
            results = self.evaluate_model(X_test_scaled, y_test)
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            # Step 6: Save results
            self.save_results()
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Results saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with exception: {e}")
            raise


def main():
    """
    Main function to run the MLP dynamic force prediction.
    """
    # Initialize predictor
    predictor = MLPDynamicForcePredictor()
    
    # Run complete pipeline with dataset-based split (L1 test, L2-L11 train)
    results = predictor.run_complete_pipeline(split_method='dataset_based')
    
    return predictor, results


if __name__ == "__main__":
    predictor, results = main()
