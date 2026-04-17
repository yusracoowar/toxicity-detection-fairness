# this imports os so i can build file paths and check whether files already exist
import os

# this imports joblib so i can load the saved vectorizer, model and threshold files
import joblib

# this imports pandas so i can create, read and update the csv review log
import pandas as pd

# this imports numpy because i need argmax and array handling for DistilBERT outputs
import numpy as np

# this imports torch because the DistilBERT model runs with pytorch
import torch

# this imports streamlit because the entire front end is built with streamlit
import streamlit as st

# this imports datetime so every saved review row gets a timestamp
from datetime import datetime

# this imports the tokenizer loader for the saved DistilBERT folder
from transformers import AutoTokenizer

# this imports the sequence classification model loader for the saved DistilBERT folder
from transformers import AutoModelForSequenceClassification

# PAGE CONFIG

# this sets the browser tab title, icon and keeps the page in wide mode
st.set_page_config(
    page_title="Toxicity Moderation Interface",
    page_icon="🛡️",
    layout="wide"
)

# CUSTOM CSS  (reference: microsoft/Streamlit_UI_Template, github.com/microsoft/Streamlit_UI_Template)

# this injects custom CSS into the page using st.markdown with unsafe_allow_html
# styling is inspired by the warm cream bento-card layout from musmentor.com
# the config.toml sets base="light" so this CSS is applied on top of a forced light base
st.markdown(
    """
    <style>

    /* this imports DM Sans from Google Fonts for the clean modern look */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

    /* this sets the main page background to a very light sky blue */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main {
        background-color: #eef4fb !important;
    }

    /* this forces the main block container to the same light blue */
    .block-container {
        background-color: #eef4fb !important;
        padding-top: 1.4rem !important;
        max-width: 1300px !important;
    }

    /* this makes vertical block wrappers transparent so the background shows through */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }

    /* this applies DM Sans across all elements with deep navy as default text */
    html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, li,
    button, input, textarea, select {
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
        color: #0a1f3c !important;
    }

    /* this removes all default Streamlit chrome for a clean app feel */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
    [data-testid="stToolbar"] { visibility: hidden; }

    /* this hides the sidebar collapse keyboard shortcut label that shows as keyboard_do... */
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarContent"] > div:first-child > div > button {
        display: none !important;
    }
    .st-emotion-cache-1cypcdb,
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* ── SIDEBAR — deep navy blue matching a school portal feel ── */
    /* this gives the sidebar a deep navy to mid-blue gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a3a6b 0%, #0f2447 100%) !important;
        border-right: none !important;
    }

    /* this forces all sidebar text to white so it reads on the deep navy */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* this styles the sidebar selectbox as a semi-transparent white pill on dark navy */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.12) !important;
        border: 1.5px solid rgba(255,255,255,0.25) !important;
        border-radius: 50px !important;
        color: #ffffff !important;
    }

    /* this keeps the selected value text white inside the navy sidebar dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #ffffff !important;
    }

    /* this styles the sidebar slider track in a bright sky blue */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
        background-color: #60a5fa !important;
    }

    /* this makes sidebar captions and small text a soft white */
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] small {
        color: rgba(255,255,255,0.75) !important;
    }

    /* ── INPUTS ── */
    /* this forces text areas to white with a soft blue border and rounded corners */
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="textarea"] textarea,
    [data-baseweb="base-input"] input {
        background-color: #ffffff !important;
        color: #0a1f3c !important;
        border: 1.5px solid #93c5fd !important;
        border-radius: 18px !important;
        font-size: 0.9rem !important;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
        padding: 0.8rem 1rem !important;
    }

    /* this styles placeholder text in a soft muted blue */
    .stTextArea textarea::placeholder {
        color: #93c5fd !important;
    }

    /* this styles the focus ring in mid blue */
    .stTextArea textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }

    /* ── RADIO BUTTONS ── */
    /* this keeps radio labels in deep navy */
    .stRadio label, .stRadio p {
        color: #0a1f3c !important;
        font-size: 0.88rem !important;
    }

    /* ── SLIDER (main area) ── */
    /* this colours the active slider track in medium blue */
    [data-testid="stSlider"] > div > div > div > div {
        background-color: #2563eb !important;
    }

    /* ── BUTTONS — royal blue pill ── */
    /* this styles all buttons as bold royal blue pills with a soft shadow */
    .stButton > button,
    .stDownloadButton > button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.5rem 1.6rem !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(37,99,235,0.25) !important;
    }

    /* this deepens the button colour and glow on hover */
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 6px 20px rgba(37,99,235,0.4) !important;
    }

    /* ── ST.CONTAINER BORDER CARDS ── */
    /* this styles all st.container cards as white rounded pills with a blue shadow */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1.5px solid #dbeafe !important;
        border-radius: 24px !important;
        box-shadow: 0 2px 12px rgba(37,99,235,0.07) !important;
    }

    /* ── DATAFRAME ── */
    /* this forces the audit table to white with rounded corners */
    [data-testid="stDataFrame"] iframe,
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border-radius: 16px !important;
    }

    /* ── ALERTS ── */
    /* this rounds all streamlit alert messages */
    [data-testid="stAlert"] {
        border-radius: 16px !important;
    }

    /* ── HEADER BANNER ── */
    /* this creates the header as a white pill card with a royal blue left accent stripe */
    .top-header {
        background-color: #ffffff;
        border: 1.5px solid #dbeafe;
        border-left: 5px solid #2563eb;
        padding: 1.1rem 1.8rem;
        border-radius: 24px;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 12px rgba(37,99,235,0.07);
    }

    /* this styles the shield emoji in the header */
    .top-header .icon {
        font-size: 2.2rem;
        line-height: 1;
    }

    /* this styles the main page title in deep navy */
    .top-header h1 {
        color: #0a1f3c !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* this styles the subtitle text in royal blue */
    .top-header p {
        color: #2563eb !important;
        font-size: 0.8rem !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.4 !important;
    }

    /* ── SECTION CARDS ── */
    /* this styles each content card as a white rounded card with a soft blue shadow */
    .section-card {
        background-color: #ffffff;
        border: 1.5px solid #dbeafe;
        border-radius: 24px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(37,99,235,0.06);
    }

    /* this styles the card heading in deep navy with a light blue underline */
    .section-card h3 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #0a1f3c !important;
        margin-top: 0 !important;
        margin-bottom: 0.7rem !important;
        border-bottom: 1.5px solid #dbeafe;
        padding-bottom: 0.45rem;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* this keeps paragraph text inside cards in a readable mid-navy */
    .section-card p {
        color: #1e40af !important;
        font-size: 0.86rem !important;
    }

    /* ── RESULT BADGES ── */
    /* this styles the toxic badge as a soft red pill */
    .badge-toxic {
        display: inline-block;
        background-color: #fef2f2;
        color: #b91c1c !important;
        border: 1.5px solid #fca5a5;
        border-radius: 50px;
        padding: 0.32rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 600;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* this styles the non-toxic badge as a soft green pill */
    .badge-safe {
        display: inline-block;
        background-color: #f0fdf4;
        color: #15803d !important;
        border: 1.5px solid #86efac;
        border-radius: 50px;
        padding: 0.32rem 1.2rem;
        font-size: 0.9rem;
        font-weight: 600;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* ── CONFIDENCE BAR ── */
    /* this styles the track of the confidence bar in light blue */
    .conf-bar-wrap {
        background-color: #dbeafe;
        border-radius: 50px;
        height: 12px;
        width: 100%;
        margin-top: 5px;
        overflow: hidden;
    }

    /* this styles the fill of the confidence bar */
    .conf-bar-fill {
        height: 12px;
        border-radius: 50px;
        transition: width 0.4s ease;
    }

    /* ── METRIC ROW ── */
    /* this styles the row of metric summary boxes */
    .metric-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }

    /* this styles each metric box with a light blue background and rounded corners */
    .metric-box {
        background-color: #eff6ff;
        border: 1.5px solid #bfdbfe;
        border-radius: 18px;
        padding: 0.6rem 1.1rem;
        min-width: 110px;
        text-align: center;
    }

    /* this styles the small uppercase label above each metric value in mid blue */
    .metric-box .m-label {
        font-size: 0.7rem !important;
        color: #2563eb !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 2px;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* this styles the large numeric metric value in deep navy */
    .metric-box .m-value {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #0a1f3c !important;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    /* ── FOOTER ── */
    /* this styles the attribution note at the bottom of the page */
    .footer-note {
        text-align: center;
        font-size: 0.74rem !important;
        color: #2563eb !important;
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1.5px solid #dbeafe;
        font-family: 'DM Sans', 'Segoe UI', Arial, sans-serif !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# FILE PATHS

# this is the folder where the saved model files are stored
SAVE_DIR = "frontend_assets"

# this is the subfolder that contains the saved DistilBERT model and tokenizer files
DISTILBERT_DIR = os.path.join(SAVE_DIR, "DistilBERT")

# this is the filename for the csv log that stores reviewer feedback
LOG_FILE = "moderation_log.csv"

# LABEL HELPERS

# this standardises prediction labels so they always display consistently across old and new rows
def clean_prediction_label(value):
    # this converts the value to text, strips spaces and lowercases it for easy comparison
    text = str(value).strip().lower()

    # this returns the standard toxic label
    if text == "toxic":
        return "Toxic"

    # this catches all previous non-toxic variants saved in different formats
    if text in ["non-toxic", "non toxic", "nontoxic"]:
        return "Non-Toxic"

    # this falls back to the original text if nothing else matches
    return str(value).strip()


# this standardises review labels so they display neatly in the audit table
def clean_review_label(value):
    # this handles empty or missing values without throwing an error
    if pd.isna(value):
        return ""

    # this replaces underscores with spaces, strips whitespace and title-cases the result
    text = str(value).replace("_", " ").strip().title()

    # this returns the cleaned label
    return text

# MODEL LOADING

# this caches all model resources so streamlit does not reload them on every interaction
@st.cache_resource
def load_assets():
    # this loads the saved tf-idf vectorizer used to turn raw text into features
    vectorizer = joblib.load(os.path.join(SAVE_DIR, "tfidf_vectorizer.joblib"))

    # this loads the saved calibrated logistic regression model
    model = joblib.load(os.path.join(SAVE_DIR, "calibrated_model.joblib"))

    # this loads the saved best threshold chosen in the notebook
    threshold = joblib.load(os.path.join(SAVE_DIR, "best_threshold.joblib"))

    # this loads the DistilBERT tokenizer from the saved folder
    distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)

    # this loads the DistilBERT classification model from the saved folder
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)

    # this switches DistilBERT to evaluation mode so dropout layers are disabled during inference
    distilbert_model.eval()

    # this returns all loaded objects as a tuple
    return vectorizer, model, threshold, distilbert_tokenizer, distilbert_model


# this unpacks all loaded model resources into named variables
vectorizer, model, saved_threshold, distilbert_tokenizer, distilbert_model = load_assets()

# LOG SETUP

# this creates the csv log file if it does not already exist on disk
if not os.path.exists(LOG_FILE):
    # this builds an empty dataframe with the correct column names for a fresh log
    empty_log = pd.DataFrame(
        columns=[
            "timestamp",
            "comment_text",
            "model_used",
            "prob_toxic",
            "threshold",
            "prediction",
            "review_label",
            "notes",
        ]
    )

    # this writes the empty dataframe to disk as the starting review log
    empty_log.to_csv(LOG_FILE, index=False)

# PREDICTION FUNCTIONS

# this runs the tf-idf baseline on one comment and returns a probability and label
def predict_comment_tfidf(text, threshold):
    # this vectorises the raw text using the saved tf-idf transformer
    text_vec = vectorizer.transform([text])

    # this gets the class 1 probability from the calibrated classifier
    prob_toxic = model.predict_proba(text_vec)[0][1]

    # this applies the threshold to produce a binary decision
    prediction = 1 if prob_toxic >= threshold else 0

    # this converts the numeric decision into a human-readable label
    label = "Toxic" if prediction == 1 else "Non-Toxic"

    # this returns the probability score and the label
    return prob_toxic, label


# this runs the saved DistilBERT model on one comment and returns a probability and label
def predict_comment_distilbert(text, threshold):
    # this tokenises the comment with truncation so it fits the model's maximum input length
    encoded = distilbert_tokenizer(
        text,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # this disables gradient computation because no training is happening here
    with torch.no_grad():
        # this passes the tokenised input through the DistilBERT model
        outputs = distilbert_model(**encoded)

        # this extracts the raw logit scores from the model output
        logits = outputs.logits

        # this converts the logits into a probability distribution across classes
        probs = torch.softmax(logits, dim=1)

        # this extracts the probability for the toxic class
        prob_toxic = probs[0, 1].item()

    # this applies the threshold to produce a binary decision
    prediction = 1 if prob_toxic >= threshold else 0

    # this converts the numeric decision to a label
    label = "Toxic" if prediction == 1 else "Non-Toxic"

    # this returns the probability and the label
    return prob_toxic, label

# LOG SAVE FUNCTION

# this appends one reviewed case to the persistent csv audit log
def save_review(comment_text, model_used, prob_toxic, threshold, prediction, review_label, notes):
    # this builds a single-row dataframe containing all fields for the review record
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "comment_text": comment_text,
                "model_used": model_used,
                "prob_toxic": prob_toxic,
                "threshold": threshold,
                "prediction": clean_prediction_label(prediction),
                "review_label": clean_review_label(review_label),
                "notes": notes,
            }
        ]
    )

    # this appends the row to the existing csv file without rewriting the header
    row.to_csv(LOG_FILE, mode="a", header=False, index=False)

# LOAD LOG FOR DISPLAY

# this reads the existing log from disk so the audit pane can display history
log_df = pd.read_csv(LOG_FILE)

# this normalises prediction labels in old rows so display is consistent
if "prediction" in log_df.columns:
    log_df["prediction"] = log_df["prediction"].apply(clean_prediction_label)

# this normalises review labels in old rows so display is consistent
if "review_label" in log_df.columns:
    log_df["review_label"] = log_df["review_label"].apply(clean_review_label)

# this counts how many logged rows were classified as toxic
toxic_count = 0
if not log_df.empty:
    toxic_count = (log_df["prediction"] == "Toxic").sum()

# this counts how many logged rows were classified as non-toxic
non_toxic_count = 0
if not log_df.empty:
    non_toxic_count = (log_df["prediction"] == "Non-Toxic").sum()

# SIDEBAR

# this builds the left sidebar with model selection and threshold controls
with st.sidebar:
    # this renders the sidebar section header
    st.markdown("## ⚙️ Settings")

    # this adds a divider line under the header
    st.markdown("---")

    # this adds a small label above the model selector
    st.markdown("**Select Model**")

    # this creates a dropdown so the reviewer can switch between the baseline and DistilBERT
    model_choice = st.selectbox(
        "Model",
        ["TF-IDF + Calibrated Classifier", "DistilBERT"],
        label_visibility="collapsed"
    )

    # this adds vertical spacing between the two controls
    st.markdown(" ")

    # this adds a label above the threshold slider
    st.markdown("**Decision Threshold**")

    # this creates the threshold slider and initialises it at the saved notebook value
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.99,
        value=float(saved_threshold),
        step=0.01,
        label_visibility="collapsed"
    )

    # this shows the currently selected threshold value below the slider
    st.markdown(
        f"<span style='font-size:0.82rem; color:rgba(255,255,255,0.8);'>Current threshold: <strong style=\"color:#ffffff;\">{threshold:.2f}</strong></span>",
        unsafe_allow_html=True
    )

    # this adds a divider and a short guidance note at the bottom of the sidebar
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.78rem; color:rgba(255,255,255,0.65);'>Raise the threshold to reduce false positives. "
        "Lower it to catch more potentially harmful content.</span>",
        unsafe_allow_html=True
    )

# PAGE HEADER BANNER

# this renders the branded navy blue header banner at the top of the main area
st.markdown(
    """
    <div class="top-header">
        <div class="icon">🛡️</div>
        <div>
            <h1>Toxicity Moderation Interface</h1>
            <p>AI Detection of Anti-Social Language in Schools and Workplaces &nbsp;·&nbsp; Birmingham City University</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# TWO-COLUMN LAYOUT

# this creates the main two-column layout: left for the review workflow, right for the audit pane
main_col, audit_col = st.columns([1.85, 1], gap="large")

# LEFT COLUMN — REVIEW WORKFLOW

# this starts the wider left column containing the main moderation cards
with main_col:

    # Input card 

    # this renders the section heading for the input card using the custom card style
    st.markdown(
        "<div class='section-card'>"
        "<h3>📝 Submit a Comment for Review</h3>",
        unsafe_allow_html=True
    )

    # this shows which model is currently active so the reviewer knows what they are using
    st.markdown(
        f"<span style='font-size:0.85rem; color:#5a7a9a;'>Active model: "
        f"<strong style='color:#1a3a5c;'>{model_choice}</strong></span>",
        unsafe_allow_html=True
    )

    # this uses a form so the textarea and submit button are submitted together in one action
    with st.form("prediction_form"):
        # this creates the multi-line comment input area with a school-appropriate placeholder
        comment_text = st.text_area(
            "Comment Text",
            height=160,
            placeholder="Type or paste a comment here to check whether it contains anti-social language.",
            label_visibility="collapsed"
        )

        # this renders the analyse button inside the form
        submitted = st.form_submit_button("🔍  Analyse Comment", use_container_width=True)

    # this closes the input section card div
    st.markdown("</div>", unsafe_allow_html=True)

    # Run prediction

    # this runs only when the user clicks the analyse button
    if submitted:
        # this shows a warning if the comment box was submitted empty
        if not comment_text.strip():
            st.warning("Please enter a comment before analysing.")
        else:
            # this runs the tf-idf baseline if it is selected
            if model_choice == "TF-IDF + Calibrated Classifier":
                prob_toxic, prediction_label = predict_comment_tfidf(comment_text, threshold)

            # this runs the DistilBERT model if it is selected
            elif model_choice == "DistilBERT":
                prob_toxic, prediction_label = predict_comment_distilbert(comment_text, threshold)

            # this saves the current comment into session state so it persists after the page reruns
            st.session_state["latest_comment"] = comment_text

            # this saves the toxic probability into session state
            st.session_state["latest_prob"] = prob_toxic

            # this saves the prediction label into session state
            st.session_state["latest_prediction"] = prediction_label

            # this saves the threshold used at prediction time into session state
            st.session_state["latest_threshold"] = threshold

            # this saves which model produced the result into session state
            st.session_state["latest_model"] = model_choice

    # Result card

    # this only shows the result card if a prediction has been made in this session
    if "latest_comment" in st.session_state:

        # this retrieves the stored prediction values from session state
        pred_label  = st.session_state["latest_prediction"]
        pred_prob   = st.session_state["latest_prob"]
        pred_thresh = st.session_state["latest_threshold"]
        pred_model  = st.session_state["latest_model"]

        # this picks the badge HTML based on whether the comment was flagged or not
        badge_html = (
            f"<span class='badge-toxic'>⚠️ Toxic</span>"
            if pred_label == "Toxic"
            else f"<span class='badge-safe'>✅ Non-Toxic</span>"
        )

        # this sets the colour for the confidence bar: red for toxic, green for safe
        bar_colour = "#e74c3c" if pred_label == "Toxic" else "#27ae60"

        # this calculates the percentage width for the confidence bar
        bar_width = int(pred_prob * 100)

        # this renders the prediction result card with badge, metric boxes and confidence bar
        st.markdown(
            f"""
            <div class='section-card'>
                <h3>📊 Prediction Result</h3>
                <div style='margin-bottom:0.9rem;'>{badge_html}</div>
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='m-label'>Confidence</div>
                        <div class='m-value'>{pred_prob:.4f}</div>
                    </div>
                    <div class='metric-box'>
                        <div class='m-label'>Threshold</div>
                        <div class='m-value'>{pred_thresh:.2f}</div>
                    </div>
                    <div class='metric-box'>
                        <div class='m-label'>Model</div>
                        <div class='m-value' style='font-size:0.78rem;'>{pred_model}</div>
                    </div>
                </div>
                <div style='font-size:0.8rem; color:#5a7a9a; margin-bottom:4px;'>Toxicity confidence</div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-fill' style='width:{bar_width}%; background-color:{bar_colour};'></div>
                </div>
                <div style='font-size:0.75rem; color:#8aa0b8; margin-top:4px;'>{bar_width}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Reviewer feedback card 

        # this renders the reviewer feedback section heading
        st.markdown(
            "<div class='section-card'><h3>✏️ Reviewer Feedback</h3>"
            "<p style='font-size:0.85rem; color:#5a7a9a; margin-top:-4px;'>"
            "Record whether the model decision was accurate before saving to the audit log.</p>",
            unsafe_allow_html=True
        )

        # this creates the feedback form so the review decision is saved in one step
        with st.form("review_form"):
            # this creates the review outcome radio buttons displayed horizontally
            review_label = st.radio(
                "Review Outcome",
                ["Correct", "False Positive", "False Negative", "Unclear"],
                horizontal=True,
            )

            # this creates the notes text area for the reviewer to add context
            notes = st.text_area(
                "Notes (optional)",
                height=90,
                placeholder="Add a short reason for your review decision, e.g. 'comment is sarcastic' or 'identity mention is neutral'.",
            )

            # this renders the save button for the review form
            review_submitted = st.form_submit_button("💾  Save Review Decision", use_container_width=True)

        # this closes the feedback card div
        st.markdown("</div>", unsafe_allow_html=True)

        # this saves the review to the log when the reviewer submits the feedback form
        if review_submitted:
            # this calls the save function with all relevant session and form values
            save_review(
                comment_text=st.session_state["latest_comment"],
                model_used=st.session_state["latest_model"],
                prob_toxic=st.session_state["latest_prob"],
                threshold=st.session_state["latest_threshold"],
                prediction=st.session_state["latest_prediction"],
                review_label=review_label,
                notes=notes,
            )

            # this shows a confirmation message so the reviewer knows the row was saved
            st.success("✅ Review decision saved to the audit log.")


# RIGHT COLUMN — AUDIT AND ANALYTICS PANE

# this starts the narrower right column for the audit history and summary counts
with audit_col:

    # this renders the audit pane section header
    st.markdown(
        "<div class='section-card'><h3>📋 Audit and Analytics Pane</h3>",
        unsafe_allow_html=True
    )

    # this creates three side-by-side columns for the summary count boxes
    c1, c2, c3 = st.columns(3)

    # this shows the total number of toxic decisions in the log
    with c1:
        st.markdown(
            f"<div class='metric-box'><div class='m-label'>Toxic</div>"
            f"<div class='m-value' style='color:#c0392b;'>{toxic_count}</div></div>",
            unsafe_allow_html=True
        )

    # this shows the total number of non-toxic decisions in the log
    with c2:
        st.markdown(
            f"<div class='metric-box'><div class='m-label'>Non-Toxic</div>"
            f"<div class='m-value' style='color:#1e8449;'>{non_toxic_count}</div></div>",
            unsafe_allow_html=True
        )

    # this shows the total number of rows in the log
    with c3:
        st.markdown(
            f"<div class='metric-box'><div class='m-label'>Total</div>"
            f"<div class='m-value'>{len(log_df)}</div></div>",
            unsafe_allow_html=True
        )

    # this adds a small gap after the summary boxes
    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

    # this renders the audit table if there are saved review rows
    if not log_df.empty:
        # this creates a display copy of the log so the original dataframe stays unchanged
        display_log_df = log_df.copy()

        # this maps internal column names to cleaner display names for the table
        rename_map = {
            "timestamp": "Time",
            "comment_text": "Comment",
            "model_used": "Model",
            "prob_toxic": "Probability",
            "prediction": "Decision",
            "review_label": "Review",
            "notes": "Notes",
        }

        # this applies the column rename to the display copy
        display_log_df = display_log_df.rename(columns=rename_map)

        # this builds the list of columns to show, only including ones that exist
        wanted_cols = [
            col for col in
            ["Time", "Model", "Comment", "Probability", "Decision", "Review", "Notes"]
            if col in display_log_df.columns
        ]

        # this keeps only the wanted display columns
        display_log_df = display_log_df[wanted_cols]

        # this rounds the probability column to four decimal places for a cleaner table
        if "Probability" in display_log_df.columns:
            display_log_df["Probability"] = display_log_df["Probability"].round(4)

        # this renders the audit dataframe table, expanding to fill the column width
        st.dataframe(display_log_df, use_container_width=True)

        # this encodes the full raw log as utf-8 csv bytes for the download button
        csv_bytes = log_df.to_csv(index=False).encode("utf-8")

        # this renders a download button so reviewers can export the full audit log as csv
        st.download_button(
            label="⬇️  Download Review Log (CSV)",
            data=csv_bytes,
            file_name="moderation_log.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        # this shows a placeholder message when the log has no rows yet
        st.info("No review history yet. Save a review decision to populate this panel.")

    # this closes the audit section card div
    st.markdown("</div>", unsafe_allow_html=True)


# PAGE FOOTER

# this renders a small footer note at the bottom of the page with project attribution
st.markdown(
    "<div class='footer-note'>"
    "Yusra Coowar · BSc Computer Science with Artificial Intelligence · Birmingham City University · "
    "Supervisor: Hadeel Saddany · This tool is for research and review support only and is not a live moderation system."
    "</div>",
    unsafe_allow_html=True
)
