import os

DATASET_PATH = "dataset/FoodClass/FoodClass2"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

total_images = 0


def count_images(folder):
    count = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                count += 1

    return count


print("=" * 50)
print("FOOD DATASET ANALYSIS")
print("=" * 50)

for category in ["Fruits", "Vegetables"]:

    category_path = os.path.join(DATASET_PATH, category)

    print(f"\n{category}")
    print("-" * 30)

    if not os.path.exists(category_path):
        print("Folder not found!")
        continue

    for food in sorted(os.listdir(category_path)):

        food_path = os.path.join(category_path, food)

        if not os.path.isdir(food_path):
            continue

        image_count = count_images(food_path)

        print(f"{food}: {image_count} images")

        total_images += image_count


print("\n" + "=" * 50)
print(f"TOTAL IMAGES: {total_images}")
print("=" * 50)