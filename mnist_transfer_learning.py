"""
MNIST Digit Recognition - Method 7: Transfer Learning with Pre-trained Models
Using pre-trained CNN models adapted for MNIST digit classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset for transfer learning"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Resize images from 28x28 to 32x32 for compatibility with pre-trained models
    x_train = tf.image.resize(x_train[..., np.newaxis], [32, 32])
    x_test = tf.image.resize(x_test[..., np.newaxis], [32, 32])
    
    # Convert grayscale to RGB (3 channels) for pre-trained models
    x_train = tf.repeat(x_train, 3, axis=-1)
    x_test = tf.repeat(x_test, 3, axis=-1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_vgg16_transfer_model(input_shape=(32, 32, 3), num_classes=10):
    """Create transfer learning model using VGG16"""
    print("Creating VGG16 transfer learning model...")
    
    # Load pre-trained VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_resnet50_transfer_model(input_shape=(32, 32, 3), num_classes=10):
    """Create transfer learning model using ResNet50"""
    print("Creating ResNet50 transfer learning model...")
    
    # Load pre-trained ResNet50 model
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def create_mobilenetv2_transfer_model(input_shape=(32, 32, 3), num_classes=10):
    """Create transfer learning model using MobileNetV2"""
    print("Creating MobileNetV2 transfer learning model...")
    
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def fine_tune_model(model, base_model, x_train, y_train, x_test, y_test):
    """Fine-tune the model by unfreezing some layers"""
    print("Fine-tuning model...")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 20
    
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune with fewer epochs
    history_fine = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history_fine

def train_transfer_model(model, x_train, y_train, x_test, y_test, model_name):
    """Train the transfer learning model"""
    print(f"\nTraining {model_name} model...")
    
    # Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.0001
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        f'best_{model_name.lower()}_transfer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    return history

def evaluate_transfer_model(model, x_test, y_test, model_name):
    """Evaluate the transfer learning model"""
    print(f"\nEvaluating {model_name} model...")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print(f"\nClassification Report - {model_name}:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} Transfer Learning')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_name.lower()}_transfer_confusion_matrix.png')
    plt.show()
    
    return test_accuracy, y_pred_classes

def visualize_feature_maps_transfer(model, x_test, model_name):
    """Visualize feature maps from transfer learning model"""
    print(f"Visualizing feature maps for {model_name}...")
    
    # Get intermediate layer outputs
    layer_outputs = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is the base model
            for sub_layer in layer.layers[-5:]:  # Last 5 layers of base model
                if len(sub_layer.output_shape) == 4:  # Conv layers
                    layer_outputs.append(sub_layer.output)
                    break
    
    if layer_outputs:
        activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
        
        # Get activations for a sample image
        sample_image = x_test[0:1]
        activations = activation_model.predict(sample_image)
        
        if len(activations) > 0:
            # Plot feature maps
            first_layer_activation = activations[0]
            if len(first_layer_activation.shape) == 4:
                n_features = min(16, first_layer_activation.shape[-1])
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                
                for i in range(n_features):
                    row = i // 4
                    col = i % 4
                    axes[row, col].imshow(first_layer_activation[0, :, :, i], cmap='viridis')
                    axes[row, col].axis('off')
                    axes[row, col].set_title(f'Filter {i+1}')
                
                plt.suptitle(f'{model_name} - Feature Maps')
                plt.tight_layout()
                plt.savefig(f'{model_name.lower()}_transfer_feature_maps.png')
                plt.show()

def plot_training_history_comparison(histories, model_names):
    """Plot training histories for comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'green', 'red']
    
    # Plot training accuracy
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax1.plot(history.history['accuracy'], color=colors[i], label=f'{name} Train')
        ax2.plot(history.history['val_accuracy'], color=colors[i], label=f'{name} Val')
    
    ax1.set_title('Training Accuracy Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Validation Accuracy Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot training loss
    for i, (history, name) in enumerate(zip(histories, model_names)):
        ax3.plot(history.history['loss'], color=colors[i], label=f'{name} Train')
        ax4.plot(history.history['val_loss'], color=colors[i], label=f'{name} Val')
    
    ax3.set_title('Training Loss Comparison')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('Validation Loss Comparison')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png')
    plt.show()

def compare_model_performance(results):
    """Compare performance of different transfer learning models"""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])
    plt.title('Transfer Learning Models - Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('transfer_learning_accuracy_comparison.png')
    plt.show()

def visualize_sample_predictions(x_test, y_test, predictions, model_names):
    """Visualize sample predictions from all models"""
    sample_indices = np.random.choice(len(y_test), 5, replace=False)
    
    fig, axes = plt.subplots(len(model_names), 5, figsize=(15, 3*len(model_names)))
    
    for model_idx, (model_name, y_pred) in enumerate(zip(model_names, predictions)):
        for i, idx in enumerate(sample_indices):
            if len(model_names) == 1:
                ax = axes[i]
            else:
                ax = axes[model_idx, i]
            
            # Convert back to grayscale for visualization
            img = x_test[idx][:,:,0]  # Take first channel
            
            ax.imshow(img, cmap='gray')
            true_label = np.argmax(y_test[idx])
            pred_label = y_pred[idx]
            ax.set_title(f'{model_name}\nTrue: {true_label}, Pred: {pred_label}')
            ax.axis('off')
            
            # Color the title based on correctness
            color = 'green' if true_label == pred_label else 'red'
            ax.title.set_color(color)
    
    plt.suptitle('Transfer Learning - Sample Predictions (Green=Correct, Red=Incorrect)')
    plt.tight_layout()
    plt.savefig('transfer_learning_sample_predictions.png')
    plt.show()

def main():
    """Main function"""
    print("MNIST Digit Recognition - Transfer Learning")
    print("=" * 50)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create models
    print("\nCreating transfer learning models...")
    
    # VGG16 model
    vgg_model, vgg_base = create_vgg16_transfer_model()
    print(f"VGG16 total parameters: {vgg_model.count_params():,}")
    
    # ResNet50 model
    resnet_model, resnet_base = create_resnet50_transfer_model()
    print(f"ResNet50 total parameters: {resnet_model.count_params():,}")
    
    # MobileNetV2 model
    mobilenet_model, mobilenet_base = create_mobilenetv2_transfer_model()
    print(f"MobileNetV2 total parameters: {mobilenet_model.count_params():,}")
    
    models = [vgg_model, resnet_model, mobilenet_model]
    model_names = ['VGG16', 'ResNet50', 'MobileNetV2']
    base_models = [vgg_base, resnet_base, mobilenet_base]
    
    # Train models
    print("\nTraining transfer learning models...")
    histories = []
    results = {}
    
    for model, name, base in zip(models, model_names, base_models):
        print(f"\n{'='*60}")
        print(f"Training {name} Transfer Learning Model")
        print('='*60)
        
        # Initial training
        history = train_transfer_model(model, x_train, y_train, x_test, y_test, name)
        histories.append(history)
        
        # Fine-tuning (optional - can be enabled for better results)
        # history_fine = fine_tune_model(model, base, x_train, y_train, x_test, y_test)
        
        # Evaluate model
        accuracy, predictions = evaluate_transfer_model(model, x_test, y_test, name)
        results[name] = {'accuracy': accuracy, 'predictions': predictions}
        
        # Visualize feature maps
        visualize_feature_maps_transfer(model, x_test, name)
        
        # Save model
        model.save(f'{name.lower()}_transfer_model.h5')
        print(f"{name} model saved as '{name.lower()}_transfer_model.h5'")
    
    # Compare models
    print("\n" + "="*60)
    print("COMPARING TRANSFER LEARNING MODELS")
    print("="*60)
    
    # Plot training history comparison
    plot_training_history_comparison(histories, model_names)
    
    # Compare model performance
    compare_model_performance(results)
    
    # Visualize sample predictions
    all_predictions = [results[name]['predictions'] for name in model_names]
    visualize_sample_predictions(x_test, y_test, all_predictions, model_names)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nBest Transfer Learning Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name in model_names:
        print(f"{name}: {results[name]['accuracy']:.4f}")
    
    return best_accuracy

if __name__ == "__main__":
    main()