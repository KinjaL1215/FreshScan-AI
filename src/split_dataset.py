import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataset/FoodClass/FoodClass2"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

data = []

for category in ["Fruits", "Vegetables"]:

    category_path = os.path.join(DATASET_PATH, category)

    for food in sorted(os.listdir(category_path)):

        food_path = os.path.join(category_path, food)

        if not os.path.isdir(food_path):
            continue

        for quality in ["Fresh", "Rotten"]:

            quality_path = os.path.join(food_path, quality)

            if not os.path.exists(quality_path):
                continue

            class_name = f"{food}_{quality}"

            for root, dirs, files in os.walk(quality_path):

                for file in files:

                    if file.lower().endswith(IMAGE_EXTENSIONS):

                        image_path = os.path.join(root, file)

                        data.append({
                            "image_path": image_path,
                            "label": class_name
                        })


# Create DataFrame
df = pd.DataFrame(data)

print("Total images:", len(df))
print("Total classes:", df["label"].nunique())

# ------------------------------------------------
# 80% Train + 20% Temporary
# ------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# ------------------------------------------------
# Split temporary into 10% Validation + 10% Test
# ------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Reset indexes
train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Save CSV files
os.makedirs("dataset/splits", exist_ok=True)

train_df.to_csv("dataset/splits/train.csv", index=False)
validation_df.to_csv("dataset/splits/validation.csv", index=False)
test_df.to_csv("dataset/splits/test.csv", index=False)

print("\n==============================")
print("DATASET SPLIT")
print("==============================")

print("Training images   :", len(train_df))
print("Validation images :", len(validation_df))
print("Testing images    :", len(test_df))

print("\nFiles created:")
print("dataset/splits/train.csv")
print("dataset/splits/validation.csv")
print("dataset/splits/test.csv")