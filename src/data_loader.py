import os
import pandas as pd
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_CSV = "dataset/splits/train.csv"
VAL_CSV = "dataset/splits/validation.csv"
TEST_CSV = "dataset/splits/test.csv"


# Load CSV files
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Create class names
class_names = sorted(train_df["label"].unique())

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMG_SIZE)

    # Convert pixel values from 0-255 to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_dataset(df, shuffle=False):
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
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(train_df, shuffle=True)
val_dataset = create_dataset(val_df)
test_dataset = create_dataset(test_df)


print("================================")
print("DATA LOADER")
print("================================")
print("Number of classes:", len(class_names))
print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nClasses:")

for i, class_name in enumerate(class_names):
    print(i, "->", class_name)

print("\nDatasets created successfully!")