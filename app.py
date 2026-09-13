import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/best_model.keras"
IMG_SIZE = (224, 224)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",

    "Banana_Fresh",
    "Banana_Rotten",

    "Bellpepper_Fresh",
    "Bellpepper_Rotten",

    "Bittergourd_Fresh",
    "Bittergourd_Rotten",

    "Capsicum_Fresh",
    "Capsicum_Rotten",

    "Carrot_Fresh",
    "Carrot_Rotten",

    "Cucumber_Fresh",
    "Cucumber_Rotten",

    "Mango_Fresh",
    "Mango_Rotten",

    "Okra_Fresh",
    "Okra_Rotten",

    "Orange_Fresh",
    "Orange_Rotten",

    "Potato_Fresh",
    "Potato_Rotten",

    "Strawberry_Fresh",
    "Strawberry_Rotten",

    "Tomato_Fresh",
    "Tomato_Rotten"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Food Quality Analyzer",
    page_icon="🍎",
    layout="centered"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(34, 197, 94, 0.15),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(168, 85, 247, 0.15),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(249, 115, 22, 0.12),
                transparent 30%
            ),
            #0b1120;
    }

    .block-container {
        max-width: 850px;
        padding-top: 35px;
        padding-bottom: 45px;
    }

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 850;

        background: linear-gradient(
            90deg,
            #22c55e,
            #06b6d4,
            #a855f7,
            #f97316
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 12px;
    }

    [data-testid="stImage"] {
        border-radius: 18px;
        overflow: hidden;
    }

    div[role="radiogroup"] {
        gap: 10px;
    }

    div[role="radiogroup"] label {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 8px 14px;
    }

    .stButton > button {
        width: 100%;
        height: 54px;

        border: none;
        border-radius: 14px;

        background: linear-gradient(
            90deg,
            #16a34a,
            #06b6d4,
            #7c3aed
        );

        color: white;

        font-size: 18px;
        font-weight: 750;

        box-shadow:
            0px 8px 25px rgba(6, 182, 212, 0.20);

        transition: 0.25s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0px 12px 30px rgba(124, 58, 237, 0.30);

        color: white;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 18px;

        box-shadow:
            0px 8px 25px rgba(0, 0, 0, 0.20);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    .stAlert {
        border-radius: 14px;
    }

    hr {
        border-color: #1e293b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI Food Quality Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based fruit and vegetable freshness detection'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# IMAGE SOURCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    'Select Food Image'
    '</div>',
    unsafe_allow_html=True
)


input_method = st.radio(
    "Choose image source",
    [
        "Upload Image",
        "Take Photo"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


uploaded_file = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG or PNG image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


# ============================================================
# CAMERA
# ============================================================

else:

    uploaded_file = st.camera_input(
        "Take a photo of your food"
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Could not read the selected image."
        )

        st.stop()


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        'Image Preview'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Selected Food Image",
        width="stretch"
    )

    st.write("")


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze = st.button(
        "Analyze Food",
        use_container_width=True
    )


    if analyze:

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with st.spinner(
            "Analyzing your food image..."
        ):

            # Resize
            image_resized = image.resize(
                IMG_SIZE
            )

            # Convert to NumPy
            image_array = np.array(
                image_resized
            )

            # Same preprocessing used during training
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Prediction
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]


            # =================================================
            # PREDICTED CLASS
            # =================================================

            predicted_index = int(
                np.argmax(predictions)
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]


            # =================================================
            # FOOD + QUALITY
            # =================================================

            food_name, quality = (
                predicted_class.rsplit(
                    "_",
                    1
                )
            )


            # =================================================
            # FRESH CLASS INDEX
            # =================================================

            fresh_class = (
                f"{food_name}_Fresh"
            )

            fresh_index = CLASS_NAMES.index(
                fresh_class
            )


            # =================================================
            # ROTTEN CLASS INDEX
            # =================================================

            rotten_class = (
                f"{food_name}_Rotten"
            )

            rotten_index = CLASS_NAMES.index(
                rotten_class
            )


            # =================================================
            # RAW PROBABILITIES
            # =================================================

            fresh_raw = float(
                predictions[fresh_index]
            )

            rotten_raw = float(
                predictions[rotten_index]
            )


            # =================================================
            # NORMALIZE FRESH + ROTTEN
            #
            # This makes:
            # Fresh % + Rotten % = 100%
            # for the detected food.
            # =================================================

            total = (
                fresh_raw +
                rotten_raw
            )


            if total > 0:

                fresh_percentage = float(
                    (fresh_raw / total) * 100
                )

                rotten_percentage = float(
                    (rotten_raw / total) * 100
                )

            else:

                fresh_percentage = 0.0
                rotten_percentage = 0.0


            # =================================================
            # MODEL CONFIDENCE
            # =================================================

            confidence = float(
                predictions[predicted_index] * 100
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## Analysis Result"
        )


        # ====================================================
        # FRESHNESS ANALYSIS
        # ====================================================

        st.markdown(
            "### 🥗 Freshness Analysis"
        )

        fresh_col, rotten_col = st.columns(2)


        # ====================================================
        # FRESH
        # ====================================================

        with fresh_col:

            st.metric(
                label="🟢 Fresh Probability",
                value=f"{fresh_percentage:.2f}%"
            )

            st.progress(
                float(
                    fresh_percentage / 100.0
                )
            )


        # ====================================================
        # ROTTEN
        # ====================================================

        with rotten_col:

            st.metric(
                label="🔴 Rotten Probability",
                value=f"{rotten_percentage:.2f}%"
            )

            st.progress(
                float(
                    rotten_percentage / 100.0
                )
            )


        # ====================================================
        # FRESH VS ROTTEN
        # ====================================================

        st.markdown(
            "### 📈 Fresh vs Rotten"
        )


        if fresh_percentage > rotten_percentage:

            st.success(
                f"🟢 **Fresh is more likely** "
                f"({fresh_percentage:.2f}%) "
                f"than Rotten "
                f"({rotten_percentage:.2f}%)."
            )

        elif rotten_percentage > fresh_percentage:

            st.error(
                f"🔴 **Rotten is more likely** "
                f"({rotten_percentage:.2f}%) "
                f"than Fresh "
                f"({fresh_percentage:.2f}%)."
            )

        else:

            st.warning(
                "⚖️ Fresh and Rotten have equal probability."
            )


        # ====================================================
        # FINAL PREDICTION
        # ====================================================

        st.markdown(
            "### 🔍 Final Prediction"
        )


        if quality == "Fresh":

            st.success(
                f"**{food_name} looks Fresh & Good!**\n\n"
                f"The AI predicts **Fresh** with "
                f"**{fresh_percentage:.2f}% probability**."
            )

        else:

            st.error(
                f"**{food_name} looks Rotten / Spoiled!**\n\n"
                f"The AI predicts **Rotten** with "
                f"**{rotten_percentage:.2f}% probability**."
            )


        # ====================================================
        # AI CONFIDENCE
        # ====================================================

        st.markdown(
            "### 🎯 AI Confidence"
        )


        confidence_value = float(
            min(
                max(
                    confidence / 100.0,
                    0.0
                ),
                1.0
            )
        )


        st.progress(
            confidence_value
        )


        st.write(
            f"**{confidence:.2f}% confidence**"
        )


        # ====================================================
        # CONFIDENCE MESSAGE
        # ====================================================

        if confidence >= 90:

            st.success(
                "⭐ **Very high confidence** — "
                "the image strongly matches the predicted class."
            )

        elif confidence >= 75:

            st.info(
                "👍 **Good confidence** — "
                "the image reasonably matches the predicted class."
            )

        elif confidence >= 50:

            st.warning(
                "⚠️ **Moderate confidence** — "
                "try a clearer image with good lighting."
            )

        else:

            st.warning(
                "⚠️ **Low confidence** — "
                "try another clearer image."
            )


# ============================================================
# NO IMAGE
# ============================================================

else:

    st.info(
        "Upload an image or take a photo "
        "of a fruit or vegetable to start the analysis."
    )