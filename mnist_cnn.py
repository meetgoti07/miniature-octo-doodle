"""
MNIST Digit Recognition - Method 2: Convolutional Neural Network (CNN)
Using TensorFlow/Keras with convolutional layers for better image processing
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset for CNN"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape data to add channel dimension (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """Create a CNN model"""
    model = keras.Sequential([
        # First convolutional block
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Second convolutional block
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),
        
        # Third convolutional block
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),
        
        # Flatten and dense layers
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_augmentation():
    """Create data augmentation generator"""
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    return datagen

def train_cnn_model(model, x_train, y_train, x_test, y_test):
    """Train the CNN model with data augmentation"""
    # Create data augmentation
    datagen = create_data_augmentation()
    datagen.fit(x_train)
    
    # Create callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        'best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model with data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=len(x_train) // 32,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    return history

def evaluate_cnn_model(model, x_test, y_test):
    """Evaluate the CNN model and show results"""
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - CNN Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('cnn_confusion_matrix.png')
    plt.show()
    
    return test_accuracy

def visualize_feature_maps(model, x_test):
    """Visualize feature maps from convolutional layers"""
    # Get outputs from each convolutional layer
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations for a sample image
    sample_image = x_test[0:1]  # First test image
    activations = activation_model.predict(sample_image)
    
    # Plot feature maps for first conv layer
    first_layer_activation = activations[0]
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(32):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(first_layer_activation[0, :, :, i], cmap='viridis')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Filter {i+1}')
    
    plt.suptitle('Feature Maps from First Convolutional Layer')
    plt.tight_layout()
    plt.savefig('cnn_feature_maps.png')
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('CNN Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('CNN Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.show()

def main():
    """Main function"""
    print("MNIST Digit Recognition - Convolutional Neural Network")
    print("=" * 60)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model()
    model.summary()
    
    # Train model
    print("\nTraining CNN model...")
    history = train_cnn_model(model, x_train, y_train, x_test, y_test)
    
    # Evaluate model
    print("\nEvaluating CNN model...")
    accuracy = evaluate_cnn_model(model, x_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    visualize_feature_maps(model, x_test)
    
    # Save model
    model.save('cnn_model.h5')
    print("\nModel saved as 'cnn_model.h5'")
    
    return accuracy

if __name__ == "__main__":
    main()