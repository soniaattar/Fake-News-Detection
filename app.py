# ============================================
# Fake / Misleading News Headline Detector
# Streamlit Web Application (PRODUCTION SAFE)
# ============================================

# ============================================
# STEP 1: Import required libraries
# ============================================
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# ============================================
# STEP 2: Download & load stopwords (safe)
# ============================================
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ============================================
# STEP 3: Text preprocessing function
# ============================================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS]
    return " ".join(words)

# ============================================
# STEP 4: Load trained PIPELINE (model + vectorizer)
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("fake_headline_pipeline.pkl")

model = load_model()

# ============================================
# STEP 5: Page configuration
# ============================================
st.set_page_config(
    page_title="Fake News Headline Detector",
    page_icon="📰",
    layout="centered"
)

# ============================================
# STEP 6: App Header
# ============================================
st.title("📰 Fake / Misleading News Headline Detector")
st.write(
    "This AI tool helps **students identify fake or misleading news headlines** "
    "using machine learning."
)

st.markdown("---")

# ============================================
# STEP 7: Main App Logic
# ============================================
def run_app():

    headline = st.text_input(
        "Enter a news headline:",
        placeholder="Scientists shocked as coffee cures all diseases"
    )

    check = st.button("Check Headline")

    if check:
        if headline.strip() == "":
            st.warning("⚠️ Please enter a headline.")
            return

        cleaned_text = clean_text(headline)

        prediction = model.predict([cleaned_text])[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned_text])[0]
            confidence = max(proba) * 100
        else:
            confidence = None

        st.markdown("---")

        if prediction != "FAKE":
            st.error("❌ **Fake / Misleading Headline Detected**")
            st.write(
                "This headline appears to contain exaggeration, "
                "sensational wording, or misleading claims."
            )
            if confidence is not None:
                st.write(f"🔍 **Confidence:** `{confidence:.2f}%`")
        else:
            st.success("✅ **Real Headline**")
            st.write(
                "This headline appears factual and non-misleading."
            )
            if confidence is not None:
                st.write(f"🔍 **Confidence:** `{confidence:.2f}%`")

# ============================================
# STEP 8: Run the app
# ============================================
if __name__ == "__main__":
    run_app()

# ============================================
# STEP 9: Footer
# ============================================
st.markdown("---")
st.caption(
    "⚙️ Model: TF-IDF + Linear Classifier (Pipeline) | "
    "📊 Accuracy ≈ 80–82%"
)
