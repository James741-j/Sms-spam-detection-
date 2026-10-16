"""Streamlit web app for SMS spam detection. Run: streamlit run app.py"""

import json

import streamlit as st

from src.config import MAX_TEXT_LENGTH, METRICS_PATH
from src.predict import predict_message

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

EXAMPLES = {
    "Prize scam": "WINNER!! You have won a free prize. Call 09061701461 now to claim your reward!",
    "Normal chat": "Are we still meeting at 5 pm today? Let me know.",
    "Urgent offer": "URGENT! Your mobile number has won 2000 pounds. Text CLAIM to 81010 now.",
}

if "message" not in st.session_state:
    st.session_state["message"] = ""


def _set_example(text: str) -> None:
    st.session_state["message"] = text


st.title("📩 SMS Spam Detector")
st.write(
    "Paste an SMS message below and the model will tell you whether it looks "
    "like spam. Built with a DevSecMLOps pipeline (tests, security scans, "
    "CI/CD and Docker)."
)

with st.sidebar:
    st.header("Model info")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.write(f"**Model:** {metrics['best_model_name']}")
        st.metric("F1-score", f"{metrics['f1']:.3f}")
        st.metric("Precision", f"{metrics['precision']:.3f}")
        st.metric("Recall", f"{metrics['recall']:.3f}")
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    else:
        st.info("Run `python -m src.train` to generate metrics.")

st.subheader("Try an example")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=_set_example, args=(text,), use_container_width=True)

st.text_area(
    "Message",
    key="message",
    height=140,
    max_chars=MAX_TEXT_LENGTH,
    placeholder="Type or paste an SMS message here...",
)

if st.button("Check message", type="primary"):
    try:
        result = predict_message(st.session_state["message"])
        pct = result["confidence"] * 100
        if result["label"] == "spam":
            st.error(f"🚨 SPAM  ({pct:.1f}% confidence)")
        else:
            st.success(f"✅ NOT SPAM  ({pct:.1f}% confidence)")
    except ValueError as err:
        st.warning(str(err))
    except FileNotFoundError as err:
        st.error(str(err))
