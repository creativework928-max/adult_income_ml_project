import os
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Income Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F8FAFC;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background:
        linear-gradient(
            135deg,
            #0B1F33,
            #2563EB
        );
        color: white;
        margin-bottom: 2rem;
    }

    .result {
        padding: 1.5rem;
        border-radius: 16px;
        background-color: white;
        border: 1px solid #E2E8F0;
        box-shadow:
        0 5px 20px rgba(
            15, 23, 42, 0.08
        );
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Adult Income Prediction</h1>
        <p>
        Real-time machine learning prediction
        powered by a production-style API.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API
# ============================================================

# API_URL = st.sidebar.text_input(
#     "Prediction API",
#     "http://api:8000"
# )


API_URL = st.sidebar.text_input(
    "Prediction API",
    os.getenv(
        "API_URL",
        "http://localhost:8000"
    )
).rstrip("/")


# ============================================================
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=100,
        value=35
    )

    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ]
    )

    fnlwgt = st.number_input(
        "Final Weight",
        min_value=1.0,
        value=180000.0
    )

    education = st.selectbox(
        "Education",
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ]
    )

    education_num = st.number_input(
        "Education Number",
        min_value=1.0,
        max_value=20.0,
        value=13.0
    )


with col2:

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ]
    )

    relationship = st.selectbox(
        "Relationship",
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ]
    )

    race = st.selectbox(
        "Race",
        [
            "White",
            "Black",
            "Asian-Pac-Islander",
            "Amer-Indian-Eskimo",
            "Other",
        ]
    )

    sex = st.selectbox(
        "Sex",
        [
            "Male",
            "Female",
        ]
    )


with col3:

    capital_gain = st.number_input(
        "Capital Gain",
        min_value=0.0,
        value=0.0
    )

    capital_loss = st.number_input(
        "Capital Loss",
        min_value=0.0,
        value=0.0
    )

    hours_per_week = st.number_input(
        "Hours per Week",
        min_value=1.0,
        max_value=100.0,
        value=40.0
    )

    native_country = st.text_input(
        "Native Country",
        "United-States"
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()


if st.button(
    "🚀 Predict Income",
    type="primary",
    use_container_width=True
):

    payload = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education_num": education_num,
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=15
        )

        response.raise_for_status()

        result = response.json()

        probability = (
            result["probability"]
        )

        st.markdown(
            '<div class="result">',
            unsafe_allow_html=True
        )

        if result["prediction"] == 1:

            st.success(
                "Predicted income: >50K"
            )

        else:

            st.info(
                "Predicted income: <=50K"
            )

        st.metric(
            "Probability of >50K",
            f"{probability:.2%}"
        )

        st.progress(
            probability
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    except Exception as exc:

        st.error(
            f"Prediction failed: {exc}"
        )