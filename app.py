import json
import joblib
import streamlit as st
from preprocess import preprocess

st.set_page_config(page_title="SpamShield", page_icon="🛡️")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

st.title("🛡️ SpamShield")
st.write("SMS Spam Detection using TF-IDF and Machine Learning")

sms = st.text_area("Enter an SMS message", height=150,
                   placeholder="e.g. Congratulations! You have won a free prize. Call now...")

if st.button("Spam Detection"):
    if not sms.strip():
        st.warning("Please type a message first.")
    else:
        cleaned = preprocess(sms)
        proba = model.predict_proba([cleaned])[0]
        spam_prob = proba[1]
        if spam_prob >= 0.5:
            st.error("🚨 Spam")
            conf = spam_prob
        else:
            st.success("✅ Not Spam (Ham)")
            conf = proba[0]
        st.metric("Confidence", f"{conf * 100:.1f}%")
        st.progress(float(spam_prob), text=f"Spam probability: {spam_prob * 100:.1f}%")

with st.expander("Model performance"):
    try:
        m = json.load(open("metrics.json"))
        st.write("Best model:", m["best_model"])
        st.table(m["results"])
    except FileNotFoundError:
        st.write("Run train.py first.")
