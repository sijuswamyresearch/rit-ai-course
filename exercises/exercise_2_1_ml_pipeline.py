w"""
Exercise 2.1: End-to-End ML Pipeline
Objective: Build a complete ML pipeline from data cleaning to evaluation.

Requirements:
- Exploratory Data Analysis (EDA)
- Feature engineering and selection
- Train multiple models (Linear Regression, Random Forest, XGBoost)
- Compare performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Part 1: Regression Pipeline (Housing Price Prediction)
# ============================================================

def regression_pipeline():
    """Complete ML pipeline for regression task."""
    print("=" * 60)
    print("REGRESSION PIPELINE: Housing Price Prediction")
    print("=" * 60)
    
    # Load data
    california = fetch_california_housing()
    X = california.data
    y = california.target
    
    # Create DataFrame for EDA
    df = pd.DataFrame(X, columns=california.feature_names)
    df['Price'] = y
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {california.feature_names}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # EDA
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nDescriptive Statistics:")
    print(df.describe())
    
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation analysis
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=150)
    plt.show()
    
    # Feature distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, column in enumerate(california.feature_names):
        axes[idx].hist(df[column], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(column)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    axes[8].hist(df['Price'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[8].set_title('Price (Target)')
    axes[8].set_xlabel('Value')
    axes[8].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150)
    plt.show()
    
    # Data preprocessing
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature engineering: Add polynomial features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    print(f"\nOriginal features: {X_train_scaled.shape[1]}")
    print(f"After polynomial features: {X_train_poly.shape[1]}")
    
    # Model training and evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'Linear Regression':
            model.fit(X_train_poly, y_train)
            y_pred_train = model.predict(X_train_poly)
            y_pred_test = model.predict(X_test_poly)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        if name == 'Linear Regression':
            cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print(f"  Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"  Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results.append({
            'model': name,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        })
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE comparison
    ax1.bar(results_df['model'], results_df['test_mse'], color=['blue', 'green', 'red'])
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Model Comparison - Test MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² comparison
    ax2.bar(results_df['model'], results_df['test_r2'], color=['blue', 'green', 'red'])
    ax2.set_ylabel('Test R²')
    ax2.set_title('Model Comparison - Test R²')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison_regression.png', dpi=150)
    plt.show()
    
    # Feature importance (for tree-based models)
    if hasattr(models['Random Forest'], 'feature_importances_'):
        importances = models['Random Forest'].feature_importances_
        feature_names = california.feature_names
        
        # Sort and plot
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Random Forest)")
        plt.barh(range(len(importances)), importances[indices], align='center')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.show()
    
    return results_df

# ============================================================
# Part 2: Classification Pipeline (Breast Cancer Dataset)
# ============================================================

def classification_pipeline():
    """Complete ML pipeline for classification task."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION PIPELINE: Breast Cancer Prediction")
    print("=" * 60)
    
    # Load data
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=cancer.feature_names)
    df['diagnosis'] = y
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {len(cancer.feature_names)}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # EDA
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nDescriptive Statistics:")
    print(df.describe())
    
    # Target distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax1.pie(np.bincount(y), labels=['Malignant (0)', 'Benign (1)'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Target Distribution')
    
    # Feature correlation with target
    correlation_with_target = df.corr()['diagnosis'].drop('diagnosis').sort_values()
    correlation_with_target.plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_title('Feature Correlation with Target')
    ax2.set_xlabel('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig('classification_eda.png', dpi=150)
    plt.show()
    
    # Data preprocessing
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training and evaluation
    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print(f"\n  Classification Report (Test Set):")
        print(f"  {classification_report(y_test, y_pred_test, target_names=cancer.target_names)}")
        
        results.append({
            'model': name,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'cv_acc_mean': cv_scores.mean(),
            'cv_acc_std': cv_scores.std()
        })
    
    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Visualize comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    ax1.bar(results_df['model'], results_df['test_acc'], color=['blue', 'green', 'red'])
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Model Comparison - Test Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim([0.9, 1.0])
    
    # CV comparison
    ax2.bar(results_df['model'], results_df['cv_acc_mean'], yerr=results_df['cv_acc_std'], 
            capsize=5, color=['blue', 'green', 'red'])
    ax2.set_ylabel('CV Accuracy')
    ax2.set_title('Model Comparison - CV Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig('model_comparison_classification.png', dpi=150)
    plt.show()
    
    # Confusion matrix for best model
    best_model_name = results_df.loc[results_df['test_acc'].idxmax(), 'model']
    best_model = models[best_model_name]
    
    y_pred_best = best_model.predict(X_test_scaled)
    
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    return results_df

if __name__ == "__main__":
    print("Exercise 2.1: End-to-End ML Pipeline")
    print("=" * 60)
    
    # Run regression pipeline
    regression_results = regression_pipeline()
    
    # Run classification pipeline
    classification_results = classification_pipeline()
    
    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE!")
    print("=" * 60)
    
    print("\nKey Learnings:")
    print("1. Complete ML pipeline from raw data to evaluation")
    print("2. Importance of EDA and data preprocessing")
    print("3. Model comparison using multiple metrics")
    print("4. Cross-validation for robust evaluation")
    print("5. Feature importance analysis")
    
    print("\nNext Steps:")
    print("- Experiment with different feature engineering techniques")
    print("- Try hyperparameter tuning with GridSearchCV")
    print("- Explore more advanced models")
    print("- Work with real-world datasets from Kaggle")