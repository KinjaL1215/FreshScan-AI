import tensorflow as tf
from data_loader import train_df, val_df, test_df
from data_loader import class_to_index, IMG_SIZE, BATCH_SIZE


# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name="data_augmentation")


def load_image_with_augmentation(image_path, label):
    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    # Normalize 0-255 -> 0-1
    image = tf.cast(image, tf.float32) / 255.0

    # Apply augmentation
    image = data_augmentation(image, training=True)

    return image, label


def create_augmented_dataset(df, shuffle=False):
    paths = df["image_path"].values
    labels = df["label"].map(class_to_index).values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image_with_augmentation,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# Training dataset with augmentation
train_augmented_dataset = create_augmented_dataset(
    train_df,
    shuffle=True
)

print("================================")
print("DATA AUGMENTATION")
print("================================")
print("Horizontal Flip : Enabled")
print("Random Rotation  : Enabled")
print("Random Zoom      : Enabled")
print("Random Contrast  : Enabled")
print("\nTraining augmentation dataset created successfully!")