# train_and_convert_tflite.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
import os

# Paths
train_dir = "data/phase1_split/Train"
val_dir   = "data/phase1_split/Validation"

# Parameters
image_size = (300, 300)
batch_size = 16
epochs     = 5

# ——— Data pipeline ———
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False
)

# Prefetch for performance
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

# ——— Model setup ———
base = EfficientNetB3(
    include_top=False,
    input_shape=(300, 300, 3),
    weights="imagenet"
)
base.trainable = False  # freeze feature extractor

inputs  = layers.Input(shape=(300, 300, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ——— Training ———
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ——— Save the trained Keras model ———
model.save("eczema_classifier_keras")

# ——— Convert to TFLite ———
converter = tf.lite.TFLiteConverter.from_saved_model("eczema_classifier_keras")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("saved model.tflite")
