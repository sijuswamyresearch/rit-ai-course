"""
Exercise 3.2: CNN for Image Classification
Objective: Build and train a Convolutional Neural Network.

Requirements:
- Design CNN architecture with multiple layers
- Implement data augmentation
- Train and evaluate the model
- Analyze misclassifications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# Part 1: Data Loading and Exploration
# ============================================================

def load_fashion_mnist():
    """
    Load Fashion MNIST dataset.
    
    Fashion MNIST consists of 70,000 grayscale images of clothing items
    in 10 different categories. Each image is 28x28 pixels.
    
    Classes:
    0: T-shirt/top
    1: Trouser
    2: Pullover
    3: Dress
    4: Coat
    5: Sandal
    6: Shirt
    7: Sneaker
    8: Bag
    9: Ankle boot
    """
    print("=" * 60)
    print("LOADING FASHION MNIST DATASET")
    print("=" * 60)
    
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Display sample images
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    
    for i in range(30):
        row = i // 10
        col = i % 10
        axes[row, col].imshow(X_train[i], cmap='gray')
        axes[row, col].set_title(class_names[y_train[i]])
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png', dpi=150)
    plt.show()
    
    # Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set distribution
    unique, counts = np.unique(y_train, return_counts=True)
    axes[0].bar([class_names[i] for i in unique], counts, color='skyblue', edgecolor='black')
    axes[0].set_title('Training Set Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Test set distribution
    unique, counts = np.unique(y_test, return_counts=True)
    axes[1].bar([class_names[i] for i in unique], counts, color='lightgreen', edgecolor='black')
    axes[1].set_title('Test Set Class Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150)
    plt.show()
    
    return X_train, y_train, X_test, y_test, class_names


def load_cifar10():
    """
    Load CIFAR-10 dataset.
    
    CIFAR-10 consists of 60,000 color images (32x32) in 10 classes.
    Each class has 6,000 images.
    
    Classes:
    0: airplane
    1: automobile
    2: bird
    3: cat
    4: deer
    5: dog
    6: frog
    7: horse
    8: ship
    9: truck
    """
    print("=" * 60)
    print("LOADING CIFAR-10 DATASET")
    print("=" * 60)
    
    # Load data
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image shape: {X_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Display sample images
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    
    for i in range(30):
        row = i // 10
        col = i % 10
        axes[row, col].imshow(X_train[i])
        axes[row, col].set_title(class_names[y_train[i][0]])
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150)
    plt.show()
    
    return X_train, y_train, X_test, y_test, class_names


# ============================================================
# Part 2: Data Preprocessing
# ============================================================

def preprocess_fashion_mnist(X_train, X_test):
    """
    Preprocess Fashion MNIST data for CNN.
    
    Steps:
    1. Reshape to add channel dimension (28, 28, 1)
    2. Normalize pixel values to [0, 1]
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING FASHION MNIST")
    print("=" * 60)
    
    # Reshape to add channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"Reshaped training data: {X_train.shape}")
    print(f"Reshaped test data: {X_test.shape}")
    
    # Normalize to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print("Data normalized to [0, 1]")
    
    return X_train, X_test


def preprocess_cifar10(X_train, X_test):
    """
    Preprocess CIFAR-10 data for CNN.
    
    Steps:
    1. Normalize pixel values to [0, 1]
    2. Convert labels to categorical (one-hot encoding)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING CIFAR-10")
    print("=" * 60)
    
    # Convert to float32 and normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    print(f"Data normalized to [0, 1]")
    
    return X_train, X_test


# ============================================================
# Part 3: Data Augmentation
# ============================================================

def create_data_augmenter_fashion_mnist():
    """
    Create data augmentation pipeline for Fashion MNIST.
    
    Augmentation techniques:
    - Random rotation
    - Random shift
    - Random zoom
    - Random flip (not used for fashion items as they have orientation)
    """
    datagen = ImageDataGenerator(
        rotation_range=10,      # Random rotation up to 10 degrees
        width_shift_range=0.1,  # Random horizontal shift
        height_shift_range=0.1, # Random vertical shift
        zoom_range=0.1,         # Random zoom
        fill_mode='nearest'     # Fill mode for new pixels
    )
    
    return datagen


def create_data_augmenter_cifar10():
    """
    Create data augmentation pipeline for CIFAR-10.
    
    Augmentation techniques:
    - Random rotation
    - Random flip (horizontal only, makes sense for objects)
    - Random shift
    - Random zoom
    """
    datagen = ImageDataGenerator(
        rotation_range=15,       # Random rotation up to 15 degrees
        width_shift_range=0.1,   # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
        zoom_range=0.1,          # Random zoom
        horizontal_flip=True,    # Random horizontal flip
        fill_mode='nearest'      # Fill mode for new pixels
    )
    
    return datagen


def visualize_augmentation(X_train, datagen, n_samples=9):
    """
    Visualize data augmentation effects.
    """
    print("\nVisualizing data augmentation...")
    
    fig, axes = plt.subplots(3, n_samples // 3, figsize=(12, 4))
    
    for i in range(n_samples):
        row = i // (n_samples // 3)
        col = i % (n_samples // 3)
        
        if i == 0:
            # Show original image
            axes[row, col].imshow(X_train[0], cmap='gray' if X_train.shape[-1] == 1 else None)
            axes[row, col].set_title('Original')
        else:
            # Show augmented version
            X_aug = datagen.random_transform(X_train[0:1])[0]
            axes[row, col].imshow(X_aug, cmap='gray' if X_aug.shape[-1] == 1 else None)
            axes[row, col].set_title(f'Augmented {i}')
        
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_augmentation_examples.png', dpi=150)
    plt.show()


# ============================================================
# Part 4: CNN Model Architectures
# ============================================================

def build_cnn_fashion_mnist(input_shape=(28, 28, 1), num_classes=10):
    """
    Build CNN architecture for Fashion MNIST.
    
    Architecture:
    - Multiple Conv2D + BatchNormalization + MaxPooling blocks
    - Dropout for regularization
    - Dense layers for classification
    """
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_cnn_cifar10(input_shape=(32, 32, 3), num_classes=10):
    """
    Build CNN architecture for CIFAR-10.
    
    This is a VGG-like architecture with:
    - Multiple Conv2D + BatchNormalization blocks
    - MaxPooling for downsampling
    - Dropout for regularization
    """
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth conv block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def build_simple_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a simpler CNN for comparison.
    """
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', padding='same', 
                     input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


# ============================================================
# Part 5: Model Training
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=50, batch_size=64, learning_rate=0.001,
                use_augmentation=True, datagen=None):
    """
    Train the CNN model with callbacks.
    
    Callbacks:
    - Early Stopping: Stop training if validation loss doesn't improve
    - Reduce LR on Plateau: Reduce learning rate when stuck
    - Model Checkpoint: Save best model
    """
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    if use_augmentation and datagen is not None:
        # Fit with data augmentation
        datagen.fit(X_train)
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            steps_per_epoch=len(X_train) // batch_size
        )
    else:
        # Fit without augmentation
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
    
    return history


# ============================================================
# Part 6: Model Evaluation and Visualization
# ============================================================

def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.
    """
    print("\n" + "=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    
    return history


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model on test set and generate reports.
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    return y_pred, y_pred_probs


def analyze_misclassifications(model, X_test, y_test, y_pred, class_names, n_samples=10):
    """
    Analyze and visualize misclassified samples.
    """
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    # Find misclassified samples
    misclassified_mask = y_pred != y_test.flatten()
    misclassified_indices = np.where(misclassified_mask)[0]
    
    print(f"Total misclassified: {len(misclassified_indices)} / {len(y_test)}")
    print(f"Misclassification rate: {len(misclassified_indices) / len(y_test):.4f}")
    
    # Visualize some misclassified samples
    n_show = min(n_samples, len(misclassified_indices))
    
    fig, axes = plt.subplots(2, n_show // 2, figsize=(15, 5))
    
    for i in range(n_show):
        row = i // (n_show // 2)
        col = i % (n_show // 2)
        idx = misclassified_indices[i]
        
        if X_test.shape[-1] == 1:
            axes[row, col].imshow(X_test[idx].squeeze(), cmap='gray')
        else:
            axes[row, col].imshow(X_test[idx])
        
        true_label = class_names[y_test[idx][0]]
        pred_label = class_names[y_pred[idx]]
        
        axes[row, col].set_title(f'True: {true_label}\nPred: {pred_label}', 
                                fontsize=9, color='red' if true_label != pred_label else 'green')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('misclassifications.png', dpi=150)
    plt.show()
    
    # Analyze which classes are most confused
    misclass_true = y_test[misclassified_indices].flatten()
    misclass_pred = y_pred[misclassified_indices]
    
    confusion_pairs = {}
    for true, pred in zip(misclass_true, misclass_pred):
        pair = (class_names[true], class_names[pred])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost common misclassifications:")
    for (true, pred), count in sorted_pairs[:10]:
        print(f"  {true} -> {pred}: {count} times")


# ============================================================
# Part 7: Feature Visualization
# ============================================================

def visualize_feature_maps(model, X_sample, layer_indices=None):
    """
    Visualize feature maps from different convolutional layers.
    """
    print("\n" + "=" * 60)
    print("FEATURE MAP VISUALIZATION")
    print("=" * 60)
    
    # Get convolutional layers
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    
    if layer_indices is None:
        layer_indices = [0, 1, 2]  # Visualize first 3 conv layers
    
    # Create model that outputs intermediate activations
    layer_outputs = [conv_layers[i].output for i in layer_indices]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(X_sample[0:1], verbose=0)
    
    # Visualize
    for layer_idx, activation in zip(layer_indices, activations):
        n_filters = activation.shape[-1]
        
        # Show up to 8 filters
        n_show = min(8, n_filters)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 4))
        fig.suptitle(f'Conv Layer {layer_idx + 1} Feature Maps', fontsize=14)
        
        for i in range(n_show):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(activation[0, :, :, i], cmap='viridis')
            axes[row, col].set_title(f'Filter {i + 1}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_show, 8):
            row = i // 4
            col = i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'feature_maps_layer_{layer_idx + 1}.png', dpi=150)
        plt.show()


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Exercise 3.2: CNN for Image Classification")
    print("=" * 60)
    
    # Choose dataset: 'fashion_mnist' or 'cifar10'
    DATASET = 'fashion_mnist'
    
    if DATASET == 'fashion_mnist':
        # Load Fashion MNIST
        X_train, y_train, X_test, y_test, class_names = load_fashion_mnist()
        
        # Preprocess
        X_train, X_test = preprocess_fashion_mnist(X_train, X_test)
        
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        # Create data augmenter
        datagen = create_data_augmenter_fashion_mnist()
        
        # Build model
        model = build_cnn_fashion_mnist()
        
    else:  # CIFAR-10
        # Load CIFAR-10
        X_train, y_train, X_test, y_test, class_names = load_cifar10()
        
        # Preprocess
        X_train, X_test = preprocess_cifar10(X_train, X_test)
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)
        y_train = y_train_cat
        y_test = y_test_cat
        
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        # Create data augmenter
        datagen = create_data_augmenter_cifar10()
        
        # Build model
        model = build_cnn_cifar10()
    
    # Visualize augmentation
    visualize_augmentation(X_train, datagen)
    
    # Train model
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=50, batch_size=64, learning_rate=0.001,
        use_augmentation=True, datagen=datagen
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    y_pred, y_pred_probs = evaluate_model(model, X_test, y_test, class_names)
    
    # Analyze misclassifications
    analyze_misclassifications(model, X_test, y_test, y_pred, class_names)
    
    # Visualize feature maps
    visualize_feature_maps(model, X_test[:10])
    
    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE!")
    print("=" * 60)
    
    print("\nKey Learnings:")
    print("1. CNN architecture design with multiple convolutional layers")
    print("2. Data augmentation techniques for improving generalization")
    print("3. Batch normalization for training stability")
    print("4. Dropout for regularization")
    print("5. Model evaluation with confusion matrix and classification report")
    print("6. Feature map visualization")
    print("7. Misclassification analysis")
    
    print("\nNext Steps:")
    print("- Try transfer learning with pre-trained models (VGG, ResNet)")
    print("- Implement more advanced architectures (ResNet, EfficientNet)")
    print("- Experiment with different data augmentation strategies")
    print("- Use learning rate scheduling")
    print("- Try mixed precision training for faster computation")