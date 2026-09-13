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
# FOOD EMOJIS
# ============================================================

FOOD_EMOJIS = {
    "Apple": "🍎",
    "Banana": "🍌",
    "Bellpepper": "🫑",
    "Bittergourd": "🥒",
    "Capsicum": "🫑",
    "Carrot": "🥕",
    "Cucumber": "🥒",
    "Mango": "🥭",
    "Okra": "🌱",
    "Orange": "🍊",
    "Potato": "🥔",
    "Strawberry": "🍓",
    "Tomato": "🍅"
}


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

    /* ========================================
       MAIN BACKGROUND
    ======================================== */

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


    /* ========================================
       MAIN CONTAINER
    ======================================== */

    .block-container {
        max-width: 850px;
        padding-top: 35px;
        padding-bottom: 45px;
    }


    /* ========================================
       HEADER
    ======================================== */

    .header-icons {
        text-align: center;
        font-size: 38px;
        margin-bottom: 10px;
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

        margin-bottom: 5px;
    }


    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 17px;
        margin-bottom: 30px;
    }


    /* ========================================
       SECTION TITLE
    ======================================== */

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* ========================================
       UPLOAD BOX
    ======================================== */

    [data-testid="stFileUploader"] {
        background: #111827;
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 12px;
    }


    /* ========================================
       IMAGE
    ======================================== */

    [data-testid="stImage"] {
        border-radius: 18px;
        overflow: hidden;
    }


    /* ========================================
       RADIO BUTTONS
    ======================================== */

    div[role="radiogroup"] {
        gap: 10px;
    }

    div[role="radiogroup"] label {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 8px 14px;
    }


    /* ========================================
       ANALYZE BUTTON
    ======================================== */

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


    /* ========================================
       METRIC CARDS
    ======================================== */

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


    /* ========================================
       ALERTS
    ======================================== */

    .stAlert {
        border-radius: 14px;
    }


    /* ========================================
       FOOTER
    ======================================== */

    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 13px;
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
    '<div class="header-icons">'
    '🍎 🥕 🍊 🍅 🥔 🍓'
    '</div>',
    unsafe_allow_html=True
)

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
        "❌ Unable to load the trained model."
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
    '📷 Select Food Image'
    '</div>',
    unsafe_allow_html=True
)


input_method = st.radio(
    "Choose image source",
    [
        "📁 Upload Image",
        "📸 Take Photo"
    ],
    horizontal=True,
    label_visibility="collapsed"
)


uploaded_file = None


# ============================================================
# UPLOAD IMAGE
# ============================================================

if input_method == "📁 Upload Image":

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
            "❌ Could not read the selected image."
        )

        st.stop()


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🖼️ Image Preview'
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
        "🔍 Analyze Food",
        use_container_width=True
    )


    if analyze:

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        with st.spinner(
            "🤖 AI is analyzing your food image..."
        ):

            # Resize image
            image_resized = image.resize(
                IMG_SIZE
            )

            # Convert image to NumPy
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

            # Model prediction
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
            # FIND FRESH CLASS
            # =================================================

            fresh_class = (
                f"{food_name}_Fresh"
            )

            fresh_index = CLASS_NAMES.index(
                fresh_class
            )


            # =================================================
            # FIND ROTTEN CLASS
            # =================================================

            rotten_class = (
                f"{food_name}_Rotten"
            )

            rotten_index = CLASS_NAMES.index(
                rotten_class
            )


            # =================================================
            # FRESH PROBABILITY
            # =================================================

            fresh_percentage = float(
                predictions[fresh_index] * 100
            )


            # =================================================
            # ROTTEN PROBABILITY
            # =================================================

            rotten_percentage = float(
                predictions[rotten_index] * 100
            )


            # =================================================
            # FINAL CONFIDENCE
            # =================================================

            confidence = float(
                predictions[predicted_index] * 100
            )


        # ====================================================
        # ANALYSIS RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## 📊 Analysis Result"
        )

        st.write("")


        # ====================================================
        # FOOD DETECTED
        # ====================================================

        food_icon = FOOD_EMOJIS.get(
            food_name,
            "🍎"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                label=f"{food_icon} Food Detected",
                value=food_name
            )


        with col2:

            if quality == "Fresh":

                quality_icon = "🟢"

            else:

                quality_icon = "🔴"


            st.metric(
                label=f"{quality_icon} Quality",
                value=quality
            )


        st.write("")


        # ====================================================
        # FOOD EMOJI
        # ====================================================

        st.markdown(
            f"""
            <div style="
                text-align:center;
                font-size:65px;
                margin:15px 0;
            ">
                {food_icon}
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # FRESH VS ROTTEN PERCENTAGES
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


        st.write("")


        # ====================================================
        # VISUAL COMPARISON
        # ====================================================

        st.markdown(
            "### 📈 Fresh vs Rotten"
        )


        if fresh_percentage > rotten_percentage:

            st.success(
                f"🟢 **Fresh is more likely** "
                f"({fresh_percentage:.2f}%) "
                f"than Rotten ({rotten_percentage:.2f}%)."
            )

        elif rotten_percentage > fresh_percentage:

            st.error(
                f"🔴 **Rotten is more likely** "
                f"({rotten_percentage:.2f}%) "
                f"than Fresh ({fresh_percentage:.2f}%)."
            )

        else:

            st.warning(
                "⚖️ The model gives equal probability "
                "to Fresh and Rotten."
            )


        # ====================================================
        # FINAL RESULT
        # ====================================================

        st.markdown(
            "### 🔍 Final Prediction"
        )


        if quality == "Fresh":

            st.success(
                f"🟢 **{food_name} looks Fresh & Good!**\n\n"
                f"😊 The AI predicts **Fresh** with "
                f"**{fresh_percentage:.2f}% probability**."
            )

            st.info(
                "🥗 **Condition:** Fresh-looking\n\n"
                "✨ The image appears visually similar "
                "to fresh food samples from the training dataset."
            )

        else:

            st.error(
                f"🔴 **{food_name} looks Rotten / Spoiled!**\n\n"
                f"⚠️ The AI predicts **Rotten** with "
                f"**{rotten_percentage:.2f}% probability**."
            )

            st.warning(
                "🚫 **Recommendation:** Avoid consuming "
                "this item if it also has visible spoilage, "
                "mold, bad smell, unusual texture, or "
                "discoloration."
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
        # CONFIDENCE INTERPRETATION
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
                "try a clearer image with good lighting "
                "for a better prediction."
            )

        else:

            st.warning(
                "⚠️ **Low confidence** — "
                "the model is not very certain about "
                "this prediction. Try another clearer image."
            )


        # ====================================================
        # SIMPLE SUMMARY
        # ====================================================

        st.markdown(
            "### 📝 Simple Summary"
        )


        if quality == "Fresh":

            st.write(
                f"{food_icon} **Food:** {food_name}\n\n"
                f"🟢 **Fresh:** {fresh_percentage:.2f}%\n\n"
                f"🔴 **Rotten:** {rotten_percentage:.2f}%\n\n"
                f"😊 **Result:** This image is more likely "
                f"to be a fresh {food_name}.\n\n"
                f"🎯 **AI Confidence:** {confidence:.2f}%"
            )

        else:

            st.write(
                f"{food_icon} **Food:** {food_name}\n\n"
                f"🟢 **Fresh:** {fresh_percentage:.2f}%\n\n"
                f"🔴 **Rotten:** {rotten_percentage:.2f}%\n\n"
                f"⚠️ **Result:** This image is more likely "
                f"to be a rotten {food_name}.\n\n"
                f"🎯 **AI Confidence:** {confidence:.2f}%"
            )


        # ====================================================
        # IMPORTANT NOTE
        # ====================================================

        st.info(
            "💡 **Important:** These percentages represent "
            "the model's visual classification probabilities. "
            "They are not a laboratory measurement of food "
            "freshness or safety."
        )


# ============================================================
# NO IMAGE MESSAGE
# ============================================================

else:

    st.info(
        "👆 Upload an image or take a photo "
        "of a fruit or vegetable to start the analysis."
    )


# ============================================================
# SUPPORTED FOODS
# ============================================================

with st.expander(
    "🥗 Supported Fruits & Vegetables"
):

    st.write(
        "🍎 Apple  •  🍌 Banana  •  🥭 Mango  •  "
        "🍊 Orange  •  🍓 Strawberry"
    )

    st.write(
        "🫑 Bellpepper  •  🫑 Capsicum  •  "
        "🥒 Bittergourd  •  🥕 Carrot"
    )

    st.write(
        "🥒 Cucumber  •  🌱 Okra  •  🥔 Potato  •  "
        "🍅 Tomato"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer-text">'
    '🍎 AI Food Quality Analyzer • '
    'MobileNetV2 Deep Learning Model'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer-text">'
    'Freshness Classification • Fruits & Vegetables'
    '</div>',
    unsafe_allow_html=True
)