
import streamlit as st
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.joblib"

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)

st.title("📩 Email Spam Classification")
st.write(
    "Machine Learning Based Spam Message Detection System"
)

st.divider()

model = load_model()

if model is None:
    st.warning(
        "Model not found. Please train the model first."
    )
    st.code("python src/train_model.py")

else:
    st.subheader("Enter Your Message")

    message = st.text_area(
        "Type your SMS or email message here:",
        placeholder="Example: Congratulations! You won a free prize..."
    )

    if st.button("Check Message", type="primary"):

        if not message.strip():
            st.warning("Please enter a message first.")

        else:
            prediction = model.predict([message])[0]

            probabilities = model.predict_proba([message])[0]
            confidence = max(probabilities) * 100

            if prediction == "spam":
                st.error("🚨 This message is SPAM!")

            else:
                st.success("✅ This message is HAM (NOT SPAM).")

            st.write(
                f"Prediction Confidence: {confidence:.2f}%"
            )

st.divider()

st.caption(
    "Developed using Python, Scikit-learn and Streamlit."
)
