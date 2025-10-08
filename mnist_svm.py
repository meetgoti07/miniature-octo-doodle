"""
MNIST Digit Recognition - Method 4: Support Vector Machine (SVM)
Using scikit-learn SVM with different kernels and feature engineering
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

def load_mnist_data():
    """Load MNIST dataset using sklearn"""
    print("Loading MNIST dataset...")
    # Load dataset
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use a subset for faster computation (you can increase this for full dataset)
    # For demo purposes, using 10000 samples
    X = X[:10000]
    y = y[:10000]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def preprocess_data(X, y, test_size=0.2):
    """Preprocess the data"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def apply_pca(X_train, X_test, n_components=0.95):
    """Apply PCA for dimensionality reduction"""
    print(f"Applying PCA with {n_components} variance ratio...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Original dimensions: {X_train.shape[1]}")
    print(f"Reduced dimensions: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_train_pca, X_test_pca, pca

def train_svm_linear(X_train, y_train, X_test, y_test):
    """Train SVM with linear kernel"""
    print("\n" + "="*50)
    print("Training SVM with Linear Kernel")
    print("="*50)
    
    # Grid search for best parameters
    param_grid = {'C': [0.1, 1, 10, 100]}
    
    svm_linear = svm.SVC(kernel='linear', random_state=42)
    grid_search = GridSearchCV(svm_linear, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return grid_search.best_estimator_, y_pred, accuracy

def train_svm_rbf(X_train, y_train, X_test, y_test):
    """Train SVM with RBF kernel"""
    print("\n" + "="*50)
    print("Training SVM with RBF Kernel")
    print("="*50)
    
    # Grid search for best parameters
    param_grid = {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    svm_rbf = svm.SVC(kernel='rbf', random_state=42)
    grid_search = GridSearchCV(svm_rbf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return grid_search.best_estimator_, y_pred, accuracy

def train_svm_polynomial(X_train, y_train, X_test, y_test):
    """Train SVM with polynomial kernel"""
    print("\n" + "="*50)
    print("Training SVM with Polynomial Kernel")
    print("="*50)
    
    # Grid search for best parameters
    param_grid = {
        'C': [1, 10],
        'degree': [2, 3],
        'gamma': ['scale', 'auto']
    }
    
    svm_poly = svm.SVC(kernel='poly', random_state=42)
    grid_search = GridSearchCV(svm_poly, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")
    
    return grid_search.best_estimator_, y_pred, accuracy

def evaluate_model(y_test, y_pred, model_name):
    """Evaluate model performance"""
    print(f"\n{model_name} - Detailed Evaluation:")
    print("-" * 40)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'svm_{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.show()
    
    return cm

def visualize_sample_predictions(X_test, y_test, y_pred, model_name, scaler=None, pca=None):
    """Visualize some sample predictions"""
    # Get some sample indices
    sample_indices = np.random.choice(len(y_test), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, idx in enumerate(sample_indices):
        row = i // 5
        col = i % 5
        
        # Reconstruct the original image
        if pca is not None and scaler is not None:
            # Inverse transform from PCA and scaling
            X_reconstructed = pca.inverse_transform(X_test[idx:idx+1])
            X_original = scaler.inverse_transform(X_reconstructed)
        else:
            X_original = X_test[idx:idx+1]
        
        # Reshape to 28x28
        img = X_original.reshape(28, 28)
        
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'True: {y_test.iloc[idx]}, Pred: {y_pred[idx]}')
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        color = 'green' if y_test.iloc[idx] == y_pred[idx] else 'red'
        axes[row, col].title.set_color(color)
    
    plt.suptitle(f'{model_name} - Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.savefig(f'svm_{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.show()

def compare_models(results):
    """Compare different SVM models"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title('SVM Model Comparison')
    plt.xlabel('SVM Kernel')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('svm_model_comparison.png')
    plt.show()

def visualize_pca_components(pca, n_components=16):
    """Visualize principal components"""
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(min(n_components, pca.n_components_)):
        row = i // 4
        col = i % 4
        
        # Reshape component to 28x28
        component = pca.components_[i].reshape(28, 28)
        
        axes[row, col].imshow(component, cmap='RdBu_r')
        axes[row, col].set_title(f'PC {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('Principal Components (First 16)')
    plt.tight_layout()
    plt.savefig('svm_pca_components.png')
    plt.show()

def main():
    """Main function"""
    print("MNIST Digit Recognition - Support Vector Machine")
    print("=" * 55)
    
    # Load data
    X, y = load_mnist_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Apply PCA for dimensionality reduction
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=0.95)
    
    # Visualize PCA components
    visualize_pca_components(pca)
    
    # Store results
    results = {}
    
    # Train different SVM models
    print("\nTraining SVM models...")
    
    # Linear SVM
    model_linear, pred_linear, acc_linear = train_svm_linear(
        X_train_pca, y_train, X_test_pca, y_test
    )
    results['Linear'] = {'model': model_linear, 'predictions': pred_linear, 'accuracy': acc_linear}
    
    # RBF SVM
    model_rbf, pred_rbf, acc_rbf = train_svm_rbf(
        X_train_pca, y_train, X_test_pca, y_test
    )
    results['RBF'] = {'model': model_rbf, 'predictions': pred_rbf, 'accuracy': acc_rbf}
    
    # Polynomial SVM
    model_poly, pred_poly, acc_poly = train_svm_polynomial(
        X_train_pca, y_train, X_test_pca, y_test
    )
    results['Polynomial'] = {'model': model_poly, 'predictions': pred_poly, 'accuracy': acc_poly}
    
    # Evaluate models
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        evaluate_model(y_test, result['predictions'], f"SVM {model_name}")
        visualize_sample_predictions(
            X_test_pca, y_test, result['predictions'], 
            f"SVM {model_name}", scaler, pca
        )
    
    # Compare models
    compare_models(results)
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nBest Model: SVM {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Save models and preprocessors
    joblib.dump(best_model, f'svm_{best_model_name.lower()}_model.pkl')
    joblib.dump(scaler, 'svm_scaler.pkl')
    joblib.dump(pca, 'svm_pca.pkl')
    
    print(f"\nBest model saved as 'svm_{best_model_name.lower()}_model.pkl'")
    print("Scaler saved as 'svm_scaler.pkl'")
    print("PCA transformer saved as 'svm_pca.pkl'")
    
    return best_accuracy

if __name__ == "__main__":
    main()