import os

DATASET_PATH = "dataset/FoodClass/FoodClass2"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

classes = []
total_images = 0


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

            image_count = 0

            for root, dirs, files in os.walk(quality_path):

                for file in files:

                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        image_count += 1

            classes.append(class_name)
            total_images += image_count

            print(f"{class_name:<25} {image_count} images")


print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Total classes : {len(classes)}")
print(f"Total images  : {total_images}")

print("\nClasses:")

for i, class_name in enumerate(classes):
    print(f"{i:2} -> {class_name}")