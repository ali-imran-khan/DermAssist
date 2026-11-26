import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Set dataset paths
train_dir = "data/phase1/Train"
test_dir = "data/phase1/Test"

# Image size and batch size
img_size = (300, 300)
batch_size = 32

# Load datasets
train_dataset = image_dataset_from_directory(
    train_dir, 
    image_size=img_size, 
    batch_size=batch_size, 
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    test_dir, 
    image_size=img_size, 
    batch_size=batch_size, 
    shuffle=False
)

# Get class names and label counts
class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"Classes: {class_names}")

# Compute class weights to handle imbalance
labels = np.concatenate([y for _, y in train_dataset], axis=0)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print(f"Class weights: {class_weights}")

# Data augmentation and normalization
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# Create the base model
base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False  # Freeze base model

# Custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dense(num_classes, activation="softmax")(x)  # Multi-class classification

# Compile the model
model = Model(inputs=base_model.input, outputs=x)
model.compile(
    optimizer=Adam(learning_rate=3e-4), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

# Training
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_dataset.map(lambda x, y: (data_augmentation(x), y)),  
    validation_data=test_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# Evaluation
test_images, test_labels = [], []
for images, labels in test_dataset:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())

test_images = np.concatenate(test_images, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

pred_probs = model.predict(test_images)
pred_classes = np.argmax(pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(test_labels, pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# AUC and ROC Curve (for binary classification)
if num_classes == 2:
    y_true = test_labels
    y_probs = pred_probs[:, 1]
    
    auc_score = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    print(f"AUC Score: {auc_score:.4f}")

    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})", color="darkorange")
    plt.plot([0, 1], [0, 1], "navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Save model
model.save("efficientnet_b3_tf_classification.h5")
print("Model saved.")
