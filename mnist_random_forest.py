"""
MNIST Digit Recognition - Method 5: Random Forest
Using scikit-learn Random Forest with feature engineering and optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import time
from sklearn.datasets import fetch_openml
import pandas as pd

def load_mnist_data(sample_size=10000):
    """Load MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use a subset for faster computation
    X = X[:sample_size]
    y = y[:sample_size]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def create_pixel_features(X):
    """Create additional pixel-based features"""
    print("Creating pixel-based features...")
    
    # Convert to numpy array if pandas DataFrame
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
    
    # Reshape to 28x28 for spatial features
    images = X_array.reshape(-1, 28, 28)
    
    features = []
    
    # Original pixel values
    features.append(X_array)
    
    # Horizontal and vertical projections
    h_proj = np.sum(images, axis=1)  # Sum along rows
    v_proj = np.sum(images, axis=2)  # Sum along columns
    features.extend([h_proj, v_proj])
    
    # Center of mass
    y_coords, x_coords = np.mgrid[0:28, 0:28]
    total_mass = np.sum(images, axis=(1, 2))
    total_mass[total_mass == 0] = 1  # Avoid division by zero
    
    center_y = np.sum(images * y_coords, axis=(1, 2)) / total_mass
    center_x = np.sum(images * x_coords, axis=(1, 2)) / total_mass
    features.extend([center_y.reshape(-1, 1), center_x.reshape(-1, 1)])
    
    # Moments (spread of the image)
    var_y = np.sum(images * (y_coords - center_y.reshape(-1, 1, 1))**2, axis=(1, 2)) / total_mass
    var_x = np.sum(images * (x_coords - center_x.reshape(-1, 1, 1))**2, axis=(1, 2)) / total_mass
    features.extend([var_y.reshape(-1, 1), var_x.reshape(-1, 1)])
    
    # Concatenate all features
    X_features = np.hstack(features)
    
    print(f"Original features: {X_array.shape[1]}")
    print(f"Enhanced features: {X_features.shape[1]}")
    
    return X_features

def preprocess_data(X, y, test_size=0.2, use_pca=True):
    """Preprocess the data with optional PCA"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Create enhanced features
    X_train_features = create_pixel_features(X_train)
    X_test_features = create_pixel_features(X_test)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Optional PCA
    pca = None
    if use_pca:
        print("Applying PCA for dimensionality reduction...")
        pca = PCA(n_components=0.95)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"PCA reduced dimensions from {X_train_features.shape[1]} to {X_train_scaled.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, pca

def train_random_forest(X_train, y_train, use_grid_search=True):
    """Train Random Forest with optional hyperparameter tuning"""
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    if use_grid_search:
        # Grid search for best parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        print("Performing grid search (this may take a while)...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        return grid_search.best_estimator_
    
    else:
        # Use default parameters with some optimization
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        print("Training with default optimized parameters...")
        start_time = time.time()
        rf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
        
        return rf

def evaluate_model(model, X_test, y_test):
    """Evaluate the Random Forest model"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('random_forest_confusion_matrix.png')
    plt.show()
    
    return y_pred, accuracy

def analyze_feature_importance(model, feature_names=None):
    """Analyze and visualize feature importance"""
    print("\nAnalyzing feature importance...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    
    # Show top 20 features
    top_features = 20
    plt.subplot(2, 1, 1)
    plt.title(f'Top {top_features} Feature Importances')
    plt.bar(range(top_features), importances[indices[:top_features]])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    
    # Show all features (overview)
    plt.subplot(2, 1, 2)
    plt.title('All Feature Importances (Overview)')
    plt.plot(importances[indices])
    plt.xlabel('Feature Rank')
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.savefig('random_forest_feature_importance.png')
    plt.show()
    
    return importances

def visualize_tree_depth_analysis(model):
    """Analyze the depth of trees in the forest"""
    depths = [tree.tree_.max_depth for tree in model.estimators_]
    
    plt.figure(figsize=(10, 6))
    plt.hist(depths, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Tree Depths in Random Forest')
    plt.xlabel('Tree Depth')
    plt.ylabel('Number of Trees')
    plt.axvline(np.mean(depths), color='red', linestyle='--', 
                label=f'Mean Depth: {np.mean(depths):.1f}')
    plt.legend()
    plt.savefig('random_forest_tree_depths.png')
    plt.show()
    
    print(f"Average tree depth: {np.mean(depths):.1f}")
    print(f"Tree depth range: {min(depths)} - {max(depths)}")

def visualize_sample_predictions(X_test, y_test, y_pred, scaler=None, pca=None):
    """Visualize some sample predictions"""
    # Get some sample indices
    sample_indices = np.random.choice(len(y_test), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, idx in enumerate(sample_indices):
        row = i // 5
        col = i % 5
        
        # For visualization, we need to reconstruct the original image from features
        # This is complex with engineered features, so we'll use a simplified approach
        
        # Get original pixel data (first 784 features if available)
        if X_test.shape[1] >= 784:
            img_data = X_test[idx, :784]
        else:
            # If PCA was applied, we can't easily reconstruct
            img_data = np.random.rand(784) * 255  # Placeholder
        
        img = img_data.reshape(28, 28)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'True: {y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]}, Pred: {y_pred[idx]}')
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        true_label = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        color = 'green' if true_label == y_pred[idx] else 'red'
        axes[row, col].title.set_color(color)
    
    plt.suptitle('Random Forest - Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.savefig('random_forest_sample_predictions.png')
    plt.show()

def cross_validation_analysis(model, X_train, y_train):
    """Perform cross-validation analysis"""
    print("\nPerforming cross-validation analysis...")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Plot CV scores
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 6), cv_scores)
    plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', 
                label=f'Mean: {cv_scores.mean():.4f}')
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('random_forest_cv_scores.png')
    plt.show()
    
    return cv_scores

def main():
    """Main function"""
    print("MNIST Digit Recognition - Random Forest")
    print("=" * 45)
    
    # Load data
    X, y = load_mnist_data(sample_size=10000)  # Adjust sample size as needed
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(X, y, use_pca=True)
    
    # Train model
    model = train_random_forest(X_train, y_train, use_grid_search=False)  # Set to True for full grid search
    
    # Cross-validation analysis
    cv_scores = cross_validation_analysis(model, X_train, y_train)
    
    # Evaluate model
    y_pred, accuracy = evaluate_model(model, X_test, y_test)
    
    # Analyze feature importance
    importances = analyze_feature_importance(model)
    
    # Analyze tree depths
    visualize_tree_depth_analysis(model)
    
    # Visualize sample predictions
    visualize_sample_predictions(X_test, y_test, y_pred, scaler, pca)
    
    # Save model and preprocessors
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'random_forest_scaler.pkl')
    if pca:
        joblib.dump(pca, 'random_forest_pca.pkl')
    
    print(f"\nModel saved as 'random_forest_model.pkl'")
    print("Scaler saved as 'random_forest_scaler.pkl'")
    if pca:
        print("PCA transformer saved as 'random_forest_pca.pkl'")
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Number of features used: {X_train.shape[1]}")
    
    return accuracy

if __name__ == "__main__":
    main()