# ============================================================
# # Clinical Trial Termination Risk Explorer
# ============================================================


# ============================================================
# Step 1. Create the basic Streamlit web application
# Purpose:
# Set up the Streamlit page and load the trained model
# before creating interactive trial inputs.
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np


# Configure the webpage
st.set_page_config(
    page_title="Clinical Trial Termination Risk Explorer",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# Step 2. Load the trained XGBoost model
# Purpose:
# Load the exported model bundle so the web app can use
# the validated prediction pipeline.
# ============================================================

model_bundle = joblib.load("model_bundle.joblib")

model = model_bundle["model"]
feature_columns = model_bundle["feature_columns"]
classification_threshold = model_bundle["classification_threshold"]


# ============================================================
# Step 3. Create the webpage header
# Purpose:
# Display the title, introduction, and research notice.
# ============================================================

st.title("Clinical Trial Termination Risk Explorer")

st.write(
    "Explore the predicted risk of clinical trial termination "
    "using trial characteristics."
)

st.info(
    "This is an interactive research prototype based on "
    "historical clinical trial data."
)

st.subheader("Trial Characteristics")


# ============================================================
# Step 4. Add interactive trial inputs
# Purpose:
# Let the user enter basic clinical-trial characteristics.
# ============================================================

enrollment = st.number_input(
    "Enrollment",
    min_value=1,
    value=100,
    step=10
)

start_year = st.number_input(
    "Start Year",
    min_value=1990,
    max_value=2030,
    value=2022,
    step=1
)

phase = st.selectbox(
    "Phase",
    [
        "EARLY_PHASE1",
        "PHASE1",
        "PHASE1/PHASE2",
        "PHASE2",
        "PHASE2/PHASE3",
        "PHASE3",
        "PHASE4",
        "UNKNOWN"
    ]
)

sponsor = st.selectbox(
    "Sponsor",
    [
        "INDUSTRY",
        "NIH",
        "FED",
        "OTHER_GOV",
        "OTHER",
        "UNKNOWN"
    ]
)

intervention = st.selectbox(
    "Intervention",
    [
        "DRUG",
        "DEVICE",
        "BIOLOGICAL",
        "BEHAVIORAL",
        "PROCEDURE",
        "RADIATION",
        "DIAGNOSTIC_TEST",
        "DIETARY_SUPPLEMENT",
        "GENETIC",
        "COMBINATION_PRODUCT",
        "OTHER"
    ]
)


# ============================================================
# Step 4A. Create disease options from model features
# Purpose:
# Automatically identify the disease categories included
# in the trained prediction model.
# ============================================================

disease_options = [
    column.replace("DISEASE_", "")
    for column in feature_columns
    if column.startswith("DISEASE_")
]

disease = st.selectbox(
    "Disease",
    disease_options
)


# Prediction button
predict_button = st.button(
    "Predict Termination Risk"
)


# ============================================================
# Step 5. Generate termination-risk prediction
# Purpose:
# Convert the user inputs into the model's expected format
# and calculate the predicted probability of termination.
# ============================================================

if predict_button:

    # Start with all model input features set to 0
    input_data = {
        column: 0
        for column in feature_columns
    }

    # Continuous variables
    input_data["log_enrollment"] = np.log1p(enrollment)
    input_data["start_year"] = start_year

    # Categorical variables
    input_data["phase"] = phase
    input_data["agency_class"] = sponsor

    # Intervention indicator
    if intervention in input_data:
        input_data[intervention] = 1

    # Disease indicator
    disease_column = f"DISEASE_{disease}"

    if disease_column in input_data:
        input_data[disease_column] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    # Predict termination probability
    termination_probability = model.predict_proba(
        input_df
    )[0, 1]

    # Apply operational classification threshold
    predicted_class = int(
        termination_probability >= classification_threshold
    )

    # Display result
    st.subheader("Prediction Result")

    st.metric(
        "Predicted Termination Risk",
        f"{termination_probability:.1%}"
    )

    if predicted_class == 1:
        st.warning(
            f"Above the selected "
            f"{classification_threshold:.0%} termination-risk threshold."
        )
    else:
        st.success(
            f"Below the selected "
            f"{classification_threshold:.0%} termination-risk threshold."
        )
