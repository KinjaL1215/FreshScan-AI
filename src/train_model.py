import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from data_loader import train_df, val_df, class_names
from data_augmentation import create_augmented_dataset
from data_loader import create_dataset


# ==============================
# SETTINGS
# ==============================

EPOCHS = 10
BATCH_SIZE = 32

os.makedirs("models", exist_ok=True)


# ==============================
# DATASETS
# ==============================

print("Creating training dataset...")

train_dataset = create_augmented_dataset(
    train_df,
    shuffle=True
)

print("Creating validation dataset...")

val_dataset = create_dataset(
    val_df,
    shuffle=False
)


# ==============================
# CLASS WEIGHTS
# ==============================

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

y_train = train_df["label"].map(class_to_index).values

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}


print("\n================================")
print("CLASS WEIGHTS")
print("================================")

for class_id, weight in class_weights.items():
    print(
        f"{class_names[class_id]:25s} : {weight:.4f}"
    )


# ==============================
# MODEL
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

x = base_model(
    inputs,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="FoodQualityMobileNetV2"
)


# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==============================
# CALLBACKS
# ==============================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ==============================
# TRAINING
# ==============================

print("\n================================")
print("STARTING TRAINING")
print("================================")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ==============================
# SAVE MODEL
# ==============================

model.save(
    "models/final_model.keras"
)

print("\n================================")
print("TRAINING COMPLETE")
print("================================")

print("Best model  : models/best_model.keras")
print("Final model : models/final_model.keras")