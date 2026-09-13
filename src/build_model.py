import tensorflow as tf

IMG_SIZE = (224, 224)
NUM_CLASSES = 26


def build_model():
    # Pretrained CNN
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Initially freeze pretrained layers
    base_model.trainable = False

    # Model architecture
    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = base_model(inputs, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs,
        name="FoodQualityMobileNetV2"
    )

    return model


model = build_model()

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("================================")
print("CNN MODEL")
print("================================")

model.summary()