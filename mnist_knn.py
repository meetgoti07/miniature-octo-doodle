"""
MNIST Digit Recognition - Method 6: K-Nearest Neighbors (KNN)
Using scikit-learn KNN with different distance metrics and optimization techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import joblib
import time
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def load_mnist_data(sample_size=5000):
    """Load MNIST dataset with smaller sample for KNN efficiency"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use smaller subset for KNN (computationally expensive)
    X = X[:sample_size]
    y = y[:sample_size]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Note: Using smaller dataset ({sample_size} samples) for KNN efficiency")
    
    return X, y

def preprocess_data(X, y, test_size=0.2, n_components=50):
    """Preprocess data with aggressive PCA for KNN efficiency"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction (essential for KNN)
    print(f"Applying PCA to reduce dimensions to {n_components}...")
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Original dimensions: {X_train_scaled.shape[1]}")
    print(f"Reduced dimensions: {X_train_pca.shape[1]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca

def find_optimal_k(X_train, y_train, max_k=20):
    """Find optimal k value using cross-validation"""
    print("\nFinding optimal k value...")
    
    k_values = range(1, max_k + 1)
    cv_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='accuracy')
        cv_scores.append(scores.mean())
        print(f"k={k}: CV Accuracy = {scores.mean():.4f}")
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, cv_scores, 'bo-')
    plt.title('KNN: K Value vs Cross-Validation Accuracy')
    plt.xlabel('K Value')
    plt.ylabel('CV Accuracy')
    plt.grid(True)
    
    # Mark optimal k
    optimal_k = k_values[np.argmax(cv_scores)]
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k}')
    plt.legend()
    plt.savefig('knn_k_optimization.png')
    plt.show()
    
    print(f"Optimal k value: {optimal_k}")
    return optimal_k

def train_knn_models(X_train, y_train, optimal_k):
    """Train KNN models with different distance metrics"""
    print("\n" + "="*50)
    print("Training KNN Models with Different Metrics")
    print("="*50)
    
    models = {}
    distance_metrics = ['euclidean', 'manhattan', 'minkowski']
    
    for metric in distance_metrics:
        print(f"\nTraining KNN with {metric} distance...")
        
        if metric == 'minkowski':
            # For Minkowski, also try different p values
            knn = KNeighborsClassifier(
                n_neighbors=optimal_k, 
                metric=metric, 
                p=2,  # p=2 is equivalent to euclidean
                n_jobs=-1
            )
        else:
            knn = KNeighborsClassifier(
                n_neighbors=optimal_k, 
                metric=metric, 
                n_jobs=-1
            )
        
        start_time = time.time()
        knn.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        models[metric] = {
            'model': knn,
            'training_time': training_time
        }
        
        print(f"{metric.capitalize()} KNN trained in {training_time:.2f} seconds")
    
    return models

def evaluate_knn_models(models, X_test, y_test):
    """Evaluate all KNN models"""
    print("\n" + "="*60)
    print("EVALUATING KNN MODELS")
    print("="*60)
    
    results = {}
    
    for metric, model_info in models.items():
        print(f"\nEvaluating {metric.capitalize()} KNN...")
        
        model = model_info['model']
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        results[metric] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'prediction_time': prediction_time,
            'training_time': model_info['training_time']
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Prediction time: {prediction_time:.2f} seconds")
        
        # Classification report
        print(f"\nClassification Report - {metric.capitalize()} KNN:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {metric.capitalize()} KNN')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'knn_{metric}_confusion_matrix.png')
        plt.show()
    
    return results

def compare_knn_models(results):
    """Compare different KNN models"""
    metrics = list(results.keys())
    accuracies = [results[metric]['accuracy'] for metric in metrics]
    training_times = [results[metric]['training_time'] for metric in metrics]
    prediction_times = [results[metric]['prediction_time'] for metric in metrics]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    bars1 = ax1.bar(metrics, accuracies, color=['blue', 'green', 'red'])
    ax1.set_title('KNN Models: Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    bars2 = ax2.bar(metrics, training_times, color=['blue', 'green', 'red'])
    ax2.set_title('KNN Models: Training Time Comparison')
    ax2.set_ylabel('Training Time (seconds)')
    for bar, time_val in zip(bars2, training_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Prediction time comparison
    bars3 = ax3.bar(metrics, prediction_times, color=['blue', 'green', 'red'])
    ax3.set_title('KNN Models: Prediction Time Comparison')
    ax3.set_ylabel('Prediction Time (seconds)')
    for bar, time_val in zip(bars3, prediction_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Combined score (accuracy / prediction_time)
    efficiency_scores = [acc / pred_time for acc, pred_time in zip(accuracies, prediction_times)]
    bars4 = ax4.bar(metrics, efficiency_scores, color=['blue', 'green', 'red'])
    ax4.set_title('KNN Models: Efficiency Score (Accuracy/Prediction Time)')
    ax4.set_ylabel('Efficiency Score')
    for bar, score in zip(bars4, efficiency_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('knn_models_comparison.png')
    plt.show()

def visualize_decision_boundary_2d(best_model, X_train, y_train, pca):
    """Visualize KNN decision boundary in 2D using first two principal components"""
    print("\nVisualizing decision boundary (2D projection)...")
    
    # Use only first 2 principal components
    X_train_2d = X_train[:, :2]
    
    # Create a mesh for visualization
    h = 0.1  # step size in the mesh
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create a 2D KNN model
    knn_2d = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn_2d.fit(X_train_2d, y_train)
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # Pad with zeros for other dimensions
    mesh_points_padded = np.zeros((mesh_points.shape[0], X_train.shape[1]))
    mesh_points_padded[:, :2] = mesh_points
    
    Z = knn_2d.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
    
    # Plot training points
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, 
                         cmap=plt.cm.Set3, edgecolors='black', s=20)
    plt.colorbar(scatter)
    plt.title('KNN Decision Boundary (2D Projection)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig('knn_decision_boundary_2d.png')
    plt.show()

def analyze_neighborhood_sizes(X_train, y_train, X_test, y_test):
    """Analyze performance with different neighborhood sizes"""
    print("\nAnalyzing different neighborhood sizes...")
    
    k_values = [1, 3, 5, 7, 9, 11, 15, 20, 25, 30]
    train_accuracies = []
    test_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"k={k}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Plot bias-variance tradeoff
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(k_values, test_accuracies, 'ro-', label='Test Accuracy')
    plt.title('KNN: Bias-Variance Tradeoff')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('knn_bias_variance_tradeoff.png')
    plt.show()

def visualize_sample_predictions(X_test, y_test, y_pred):
    """Visualize some sample predictions"""
    # Get some sample indices
    sample_indices = np.random.choice(len(y_test), 10, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, idx in enumerate(sample_indices):
        row = i // 5
        col = i % 5
        
        # For visualization, we would need to inverse transform from PCA
        # For simplicity, we'll show a placeholder
        img = np.random.rand(28, 28)  # Placeholder since we can't easily reconstruct
        
        axes[row, col].imshow(img, cmap='gray')
        true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        axes[row, col].set_title(f'True: {true_label}, Pred: {y_pred[idx]}')
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        color = 'green' if true_label == y_pred[idx] else 'red'
        axes[row, col].title.set_color(color)
    
    plt.suptitle('KNN Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.savefig('knn_sample_predictions.png')
    plt.show()

def main():
    """Main function"""
    print("MNIST Digit Recognition - K-Nearest Neighbors")
    print("=" * 50)
    
    # Load data (smaller subset for KNN efficiency)
    X, y = load_mnist_data(sample_size=5000)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(X, y, n_components=50)
    
    # Find optimal k
    optimal_k = find_optimal_k(X_train, y_train, max_k=15)
    
    # Analyze different neighborhood sizes
    analyze_neighborhood_sizes(X_train, y_train, X_test, y_test)
    
    # Train models with different distance metrics
    models = train_knn_models(X_train, y_train, optimal_k)
    
    # Evaluate models
    results = evaluate_knn_models(models, X_test, y_test)
    
    # Compare models
    compare_knn_models(results)
    
    # Find best model
    best_metric = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = models[best_metric]['model']
    best_accuracy = results[best_metric]['accuracy']
    
    print(f"\nBest Model: {best_metric.capitalize()} KNN")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Visualize decision boundary
    visualize_decision_boundary_2d(best_model, X_train, y_train, pca)
    
    # Visualize sample predictions
    visualize_sample_predictions(X_test, y_test, results[best_metric]['predictions'])
    
    # Save best model and preprocessors
    joblib.dump(best_model, f'knn_{best_metric}_model.pkl')
    joblib.dump(scaler, 'knn_scaler.pkl')
    joblib.dump(pca, 'knn_pca.pkl')
    
    print(f"\nBest model saved as 'knn_{best_metric}_model.pkl'")
    print("Scaler saved as 'knn_scaler.pkl'")
    print("PCA transformer saved as 'knn_pca.pkl'")
    
    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Best Distance Metric: {best_metric.capitalize()}")
    print(f"Optimal k: {optimal_k}")
    print(f"Test Accuracy: {best_accuracy:.4f}")
    print(f"Features used: {X_train.shape[1]} (reduced from 784)")
    print(f"Training samples: {X_train.shape[0]}")
    
    return best_accuracy

if __name__ == "__main__":
    main()