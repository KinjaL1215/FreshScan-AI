import os

DATASET_PATH = "dataset/FoodClass/FoodClass2"

for category in ["Fruits", "Vegetables"]:

    category_path = os.path.join(DATASET_PATH, category)

    print("\n" + "=" * 50)
    print(category)
    print("=" * 50)

    for food in sorted(os.listdir(category_path)):

        food_path = os.path.join(category_path, food)

        if not os.path.isdir(food_path):
            continue

        print(f"\n{food}/")

        items = os.listdir(food_path)

        for item in sorted(items)[:20]:
            item_path = os.path.join(food_path, item)

            if os.path.isdir(item_path):
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")

        if len(items) > 20:
            print(f"   ... and {len(items) - 20} more")