import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import pickle
import lightgbm as lgb


# Custom HTML f title with background and text color
st.markdown("""
    <div style="background-color:#2E86C1;padding:6px;border-radius:6px">
    <h1 style="color:white;text-align:center;">Product Sales Forecasting</h1>
    </div>
    """, unsafe_allow_html=True)


def extract_features(user_input):
    # Create a DataFrame from user input
    df = pd.DataFrame([user_input],index=[0])
    
    # Extracting Year, Month, Quarter, Week, Day, Day of Week from Date
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day'] = df['Date'].dt.day
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day of Week'] = df['Date'].dt.day_name()
    
    # Label Encoding for 'Holiday' and 'Discount'
    df['Holiday'] = df['Holiday'].map({'No': 0, 'Yes': 1})
    df['Discount'] = df['Discount'].map({'No': 0, 'Yes': 1})
    
    # Convert to categorical as needed
    columns = ['Year', 'Month', 'Quarter', 'Day', 'Week', 'Day of Week']
    for col in columns:
        df[col] = df[col].astype('category')
    return df

def one_hot_encode_features(df):
    # List of features to be one-hot encoded
    categorical_features = ['Store_Type', 'Location_Type', 'Region_Code',
                            'Year', 'Month', 'Quarter', 'Week', 'Day', 'Day of Week']
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)
    
    return df

def align_features_with_model(df):
    expected_columns = [
    'Holiday', 'Discount', 'Store_Type_S2', 'Store_Type_S3',
    'Store_Type_S4', 'Location_Type_L2', 'Location_Type_L3',
    'Location_Type_L4', 'Location_Type_L5', 'Region_Code_R2',
    'Region_Code_R3', 'Region_Code_R4', 'Year_2019', 'Month_2',
    'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7', 'Month_8',
    'Month_9', 'Month_10', 'Month_11', 'Month_12', 'Quarter_2',
    'Quarter_3', 'Quarter_4', 'Week_2', 'Week_3', 'Week_4', 'Week_5',
    'Week_6', 'Week_7', 'Week_8', 'Week_9', 'Week_10', 'Week_11',
    'Week_12', 'Week_13', 'Week_14', 'Week_15', 'Week_16', 'Week_17',
    'Week_18', 'Week_19', 'Week_20', 'Week_21', 'Week_22', 'Week_23',
    'Week_24', 'Week_25', 'Week_26', 'Week_27', 'Week_28', 'Week_29',
    'Week_30', 'Week_31', 'Week_32', 'Week_33', 'Week_34', 'Week_35',
    'Week_36', 'Week_37', 'Week_38', 'Week_39', 'Week_40', 'Week_41',
    'Week_42', 'Week_43', 'Week_44', 'Week_45', 'Week_46', 'Week_47',
    'Week_48', 'Week_49', 'Week_50', 'Week_51', 'Week_52', 'Day_2',
    'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'Day_8', 'Day_9',
    'Day_10', 'Day_11', 'Day_12', 'Day_13', 'Day_14', 'Day_15',
    'Day_16', 'Day_17', 'Day_18', 'Day_19', 'Day_20', 'Day_21',
    'Day_22', 'Day_23', 'Day_24', 'Day_25', 'Day_26', 'Day_27',
    'Day_28', 'Day_29', 'Day_30', 'Day_31', 'Day of Week_Monday',
    'Day of Week_Saturday', 'Day of Week_Sunday', 'Day of Week_Thursday',
    'Day of Week_Tuesday', 'Day of Week_Wednesday'
    ]

    # Add missing columns with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure the correct column order
    df = df[expected_columns]
    
    return df

def scale_features(df, scaler):
    # Scale numerical features
    df = scaler.transform(df)
    
    return df

def make_prediction(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction


# User inputs and feature extraction
def get_user_inputs(store_df):
    st.sidebar.header('User Input Features')
    
    # Option to select Store ID or manually input the other fields
    option = st.sidebar.selectbox("Select Input Method", ("Store ID", "Store Type, Location, Region"))
    
    if option == "Store ID":
        store_id = st.sidebar.number_input('Store ID', min_value=1, max_value=365, step=1)
        # Fetch Store_Type, Location_Type, and Region_Code based on the store_id
        store_info = store_df.loc[store_id] if store_id in store_df.index else None
        
        if store_info is not None:
            st.sidebar.write(f"Store_Type: {store_info['Store_Type']}")
            st.sidebar.write(f"Location_Type: {store_info['Location_Type']}")
            st.sidebar.write(f"Region_Code: {store_info['Region_Code']}")
        else:
            st.sidebar.write("Invalid Store ID")

        store_type = store_info['Store_Type']
        location_type = store_info['Location_Type']
        region_code = store_info['Region_Code']
        
    else:
        store_id = None
        store_type = st.sidebar.selectbox('Store Type', options=['S1', 'S2', 'S3', 'S4'])
        location_type = st.sidebar.selectbox('Location Type', options=['L1', 'L2', 'L3', 'L4', 'L5'])
        region_code = st.sidebar.selectbox('Region Code', options=['R1', 'R2', 'R3', 'R4'])
    
    # Date input
    date = st.sidebar.date_input('Date', min_value=datetime(2018, 1, 1), max_value=datetime(2019, 12, 31))
    
    # Holiday input
    holiday = st.sidebar.selectbox('Holiday', options=['No', 'Yes'])
    
    # Discount input
    discount = st.sidebar.selectbox('Discount', options=['No', 'Yes'])
    
    return {
        'Store_id': store_id,
        'Store_Type': store_type,
        'Location_Type': location_type,
        'Region_Code': region_code,
        'Date': date,
        'Holiday': holiday,
        'Discount': discount
    }

# Display user inputs on the app
def display_inputs(inputs):
    if inputs['Store_id']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Store Type:** {inputs['Store_Type']}")
        with col2:
            st.write(f"**Location Type:** {inputs['Location_Type']}")
        with col3:
            st.write(f"**Region Code:** {inputs['Region_Code']}")
    else:
        pass

# Load the model and scaler
def load_resources():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('productSales_stacking_regressor.pkl', 'rb') as file:
        model = pickle.load(file)
    return scaler, model


def historical_plots(df):
    df['Date'] = pd.to_datetime(df['Date'])
    st.subheader(f'Historical Daily Sales')

    daily_sales = df.groupby(['Date'])['Sales'].sum()

    st.line_chart(daily_sales)

    col1, col2 = st.columns(2)
    with col1: 
        st.subheader('Historical Weekly Sales')
        weekly_sales = daily_sales.resample('W').sum()
        st.line_chart(weekly_sales)
    with col2: 
        st.subheader('Historical Monthly Sales')
        monthly_sales = daily_sales.resample('M').sum()
        st.line_chart(monthly_sales)

def display_prediction(prediction):
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;">
            <h2 style="color: #333333; font-size: 36px;">Predicted Sales: ${prediction:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add a button to reset or make a new prediction
    if st.button('Make Another Prediction'):
        st.experimental_rerun()


def main():
    
    df = pd.read_csv('TRAIN.csv')
    store_df = df.groupby(['Store_id'])[['Store_Type', 'Location_Type', 'Region_Code']].agg('max')

    # st.title('Sales Prediction App')
    
    user_input = get_user_inputs(store_df)

    # st.subheader('User Input')
    # st.write(user_input)
    
    # Further processing and prediction will be implemented here...
    # st.write("Feature processing and prediction logic goes here.")

    # # Get user inputs
    # user_input = get_user_inputs()
    
    # Load the scaler and model
    scaler, model = load_resources()
    
    # Extract features from user input
    extracted_features = extract_features(user_input)

    # One-Hot Encode the features
    encoded_features = one_hot_encode_features(extracted_features)
    
    # Align features with model input
    aligned_features = align_features_with_model(encoded_features)
    
    # Scale the features
    scaled_features = scale_features(aligned_features, scaler)
    
    if st.button("Predict Sales"):
        # Make the prediction
        prediction = make_prediction(scaled_features, model)
        
        # Display the prediction with formatting
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <h2 style="color: #4CAF50; text-align: center;">Predicted Sales:</h2>
                <h1 style="font-size: 50px; text-align: center; color: #333;">{prediction[0]:,.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    historical_plots(df)


if __name__ == "__main__":
    main()


