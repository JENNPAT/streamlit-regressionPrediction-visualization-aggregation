import streamlit as st
import pandas as pd
import joblib

st.title('House Price Prediction App')

# Input fields
area = st.number_input('Area (sq ft)', min_value=0)
bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value =6, step=1)
bathrooms = st.slider('Number of Bathrooms', min_value=1, max_value=4, step=1)
stories = st.slider('Number of Stories', min_value=1, max_value=4, step=1)
parking = st.slider('Parking Spaces', min_value=0, max_value=3, step=1)

# Categorical features
mainroad = st.selectbox('Main Road Access', ['yes', 'no'])
guestroom = st.selectbox('Guest Room Available', ['yes', 'no'])
basement = st.selectbox('Basement Available', ['yes', 'no'])
hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

# mapping = {'Yes': 1, 'No': 0}
# HasCrCard_num = mapping[HasCrCard]
# IsActiveMember_num = mapping[IsActiveMember]

# Create a DataFrame from the inputs
input_df = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus],
})

# Additional features
input_df['area_per_bedroom'] = input_df['area'] / input_df['bedrooms']
input_df['bathroom_per_bedroom'] = input_df['bathrooms'] / input_df['bedrooms']
input_df['good_location'] = ((input_df['mainroad'] =="yes") & (input_df['prefarea'] >= "yes")).astype(int)

input_df['luxury_house'] = (
    (input_df['bedrooms'] >= 3) & 
    (input_df['bathrooms'] >= 2) & 
    (input_df['stories'] >= 2) & 
    (input_df['parking'] >= 2) & 
    (input_df['area'] > 8000) 
).astype(int)

# Load the pre-fitted encoders, scaler and model
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Encode categorical features
encoded_features = encoder.transform(input_df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                               'airconditioning', 'prefarea', 'furnishingstatus']])

# Convert encoded features to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

# Concatenate encoded features with numerical features
preprocessed_df = pd.concat([
    input_df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'area_per_bedroom', 'bathroom_per_bedroom', 'good_location', 'luxury_house']],
    encoded_df
], axis=1)

scaled_features = scaler.transform(preprocessed_df)
preprocessed_df = pd.DataFrame(scaled_features, columns=preprocessed_df.columns)

# Add a button for prediction
if st.button('Predict'):
    prediction = model.predict(preprocessed_df)
    st.success(f'The predicted house price is: ${prediction.item():,.2f}')
