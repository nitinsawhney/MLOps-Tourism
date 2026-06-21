import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib


# Download and load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="nitinsawhney/tourism_pkg_prediction_model", 
            filename="best_tourism_prediction_model_v1.joblib"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction")

st.markdown("""
<style>
    .main-header {
        font-size: 20px;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #e6f7dc;
        border: 1px solid #a9c29b;
    }
    .warning-box {
        background-color: #f7dcc3;
        border: 1px solid #fabd84;
    }
</style>
""", unsafe_allow_html=True)

st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package 
based on their demographics and interaction data. Please enter customer data below to get the prediction.
""")

# Let's create two columns for user input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
    
    type_of_contact = st.selectbox(
        "Type of Contact",
        options=["Self Enquiry", "Company Invited"]
    )
    
    city_tier = st.selectbox(
        "City Tier",
        options=[1, 2, 3],
        help="Tier 1: Metro cities, Tier 2: Mid-sized cities, Tier 3: Smaller cities"
    )
    
    occupation = st.selectbox(
        "Occupation",
        options=["Free Lancer", "Large Business", "Salaried", "Small Business"]
    )
    
    gender = st.selectbox("Gender", options=["Male", "Female"])
    
    marital_status = st.selectbox(
        "Marital Status",
        options=["Single", "Married", "Divorced", "Unmarried"]
    )
    
    designation = st.selectbox(
        "Designation",
        options=["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    
    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        max_value=200000.0,
        value=25000.0,
        step=1000.0
    )

with col2:
    st.subheader("Customer Interaction & Preferences")
    
    duration_of_pitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0.0,
        max_value=60.0,
        value=15.0,
        step=0.5
    )
    
    number_of_persons_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )
    
    number_of_followups = st.number_input(
        "Number of Follow-ups",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )
    
    product_pitched = st.selectbox(
        "Product Pitched",
        options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    
    preferred_property_star = st.selectbox(
        "Preferred Property Star Rating",
        options=[3, 4, 5]
    )
    
    number_of_trips = st.number_input(
        "Number of Trips (per year)",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )
    
    option_values = {
        "Yes": 1,
        "No": 0
    }

    selected_option = st.selectbox(
        "Has Passport?",
        option_values.keys()
    )

    passport = option_values[selected_option]    
    
    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )
    
    car_option = st.selectbox(
        "Owns Car?",
        option_values.keys()
    )

    own_car = option_values[car_option]  
    
    number_of_children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )


# Prepare input data
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_persons_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
st.markdown("---")
if st.button("Predict Purchase Likelihood", use_container_width=True):
    if model is not None:
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("###Prediction Results")
            
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box success-box">'
                    f'<h2>There is a High Likelihood of Purchase!</h2>'
                    f'<p style="font-size: 18px;">This customer is <b>likely to purchase</b> the Wellness Tourism Package.</p>'
                    f'<p style="font-size: 16px;">Confidence: <b>{prediction_proba[1]*100:.2f}%</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.success("**Recommendation:** Prioritize follow-up with this customer!")
            else:
                st.markdown(
                    f'<div class="prediction-box warning-box">'
                    f'<h2>There is a Low Likelihood of Purchase</h2>'
                    f'<p style="font-size: 18px;">This customer is <b>unlikely to purchase</b> the Wellness Tourism Package.</p>'
                    f'<p style="font-size: 16px;">Confidence: <b>{prediction_proba[0]*100:.2f}%</b></p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.info("**Recommendation:** Consider alternative packages or additional engagement strategies.")
            
            # Show probability breakdown
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Probability of No Purchase", f"{prediction_proba[0]*100:.2f}%")
            with col_prob2:
                st.metric("Probability of Purchase", f"{prediction_proba[1]*100:.2f}%")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded. Please check the model repository.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Wellness Tourism Package Prediction System</p>
    <p>Built using Streamlit and XGBoost</p>
</div>
""", unsafe_allow_html=True)
