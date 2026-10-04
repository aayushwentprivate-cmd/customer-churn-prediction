import streamlit as st
import pandas as pd
import pickle


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict whether "
    "the customer is likely to churn."
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)

    # If pickle contains a dictionary
    if isinstance(model_data, dict):

        # Try common keys first
        for key in ["model", "best_model", "classifier", "estimator"]:
            if key in model_data:
                possible_model = model_data[key]

                if hasattr(possible_model, "predict"):
                    return possible_model

        # Otherwise search dictionary values
        for value in model_data.values():
            if hasattr(value, "predict") and hasattr(value, "predict_proba"):
                return value

        raise ValueError(
            "Could not find a trained ML model inside customer_churn_model.pkl. "
            f"Available keys: {list(model_data.keys())}"
        )

    # If pickle directly contains the model
    if hasattr(model_data, "predict"):
        return model_data

    raise ValueError(
        "customer_churn_model.pkl does not contain a valid ML model."
    )


# =========================================================
# LOAD ENCODERS
# =========================================================

@st.cache_resource
def load_encoders():

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return encoders


try:

    model = load_model()
    encoders = load_encoders()

except Exception as e:

    st.error("❌ Error loading model or encoders.")
    st.exception(e)
    st.stop()


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")


col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# Column 1
# ---------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )


# ---------------------------------------------------------
# Column 2
# ---------------------------------------------------------

with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


# ---------------------------------------------------------
# Column 3
# ---------------------------------------------------------

with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# BILLING INFORMATION
# =========================================================

st.divider()

st.header("💰 Billing Information")

billing_col1, billing_col2 = st.columns(2)

with billing_col1:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=29.85,
        step=0.01
    )

with billing_col2:

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=29.85,
        step=0.01
    )


# =========================================================
# PREDICTION
# =========================================================

st.divider()


if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    try:

        # -------------------------------------------------
        # Create input dictionary
        # -------------------------------------------------

        input_data = {

            "gender": gender,

            "SeniorCitizen": senior_citizen,

            "Partner": partner,

            "Dependents": dependents,

            "tenure": tenure,

            "PhoneService": phone_service,

            "MultipleLines": multiple_lines,

            "InternetService": internet_service,

            "OnlineSecurity": online_security,

            "OnlineBackup": online_backup,

            "DeviceProtection": device_protection,

            "TechSupport": tech_support,

            "StreamingTV": streaming_tv,

            "StreamingMovies": streaming_movies,

            "Contract": contract,

            "PaperlessBilling": paperless_billing,

            "PaymentMethod": payment_method,

            "MonthlyCharges": monthly_charges,

            "TotalCharges": total_charges
        }


        # -------------------------------------------------
        # Convert dictionary to DataFrame
        # -------------------------------------------------

        input_df = pd.DataFrame([input_data])


        # -------------------------------------------------
        # Encode categorical columns
        # -------------------------------------------------

        for column, encoder in encoders.items():

            if column in input_df.columns:

                try:

                    input_df[column] = encoder.transform(
                        input_df[column]
                    )

                except Exception as e:

                    st.error(
                        f"Encoding error in column: {column}"
                    )

                    st.exception(e)

                    st.stop()


        # -------------------------------------------------
        # Arrange columns in model's expected order
        # -------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_columns = list(model.feature_names_in_)

            input_df = input_df[expected_columns]


        # -------------------------------------------------
        # Make prediction
        # -------------------------------------------------

        prediction = model.predict(input_df)


        # -------------------------------------------------
        # Prediction probability
        # -------------------------------------------------

        if hasattr(model, "predict_proba"):

            pred_prob = model.predict_proba(input_df)

            no_churn_probability = pred_prob[0][0]

            churn_probability = pred_prob[0][1]

        else:

            no_churn_probability = None

            churn_probability = None


        # =================================================
        # DISPLAY RESULT
        # =================================================

        st.divider()

        st.header("📊 Prediction Result")


        if prediction[0] == 1:

            st.error(
                "⚠️ Customer is likely to CHURN"
            )

        else:

            st.success(
                "✅ Customer is NOT likely to churn"
            )


        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        if churn_probability is not None:

            st.subheader("Prediction Probability")

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:

                st.metric(
                    "No Churn Probability",
                    f"{no_churn_probability:.2%}"
                )

            with probability_col2:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}"
                )


            # -------------------------------------------------
            # Progress bar
            # -------------------------------------------------

            st.write("Churn Probability")

            st.progress(
                float(churn_probability)
            )


        # -------------------------------------------------
        # Show encoded input (optional)
        # -------------------------------------------------

        with st.expander("🔍 View Processed Input Data"):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.exception(e)