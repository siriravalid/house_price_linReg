import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
df = pd.read_csv('house_prices.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.drop(['street', 'city', 'statezip', 'country', 'date'], axis=1)
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
st.title("House Price Prediction(linear regression)- Siri")
bedrooms = st.number_input("Number of Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)
sqft_lot = st.number_input("Lot Area (sqft)", min_value=500, max_value=100000, value=5000)
floors = st.number_input("Number of Floors", min_value=1.0, max_value=3.0, value=1.0)
waterfront = st.selectbox("Waterfront", [0, 1])
view = st.selectbox("View", [0, 1, 2, 3, 4])
condition = st.selectbox("Condition", [1, 2, 3, 4, 5])
sqft_above = st.number_input("Above Ground Living Area (sqft)", min_value=500, max_value=10000, value=1500)
sqft_basement = st.number_input("Basement Area (sqft)", min_value=0, max_value=5000, value=500)
yr_built = st.number_input("Year Built", min_value=1900, max_value=2023, value=2000)
yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2023, value=0)
if st.button("Predict"):
    input_data = pd.DataFrame({'bedrooms': [bedrooms],'bathrooms': [bathrooms],'sqft_living': [sqft_living],'sqft_lot': [sqft_lot],'floors': [floors],'waterfront': [waterfront],'view': [view],'condition': [condition],'sqft_above': [sqft_above],'sqft_basement': [sqft_basement],'yr_built': [yr_built],'yr_renovated': [yr_renovated]})
    predicted_price = model.predict(input_data)[0]
    st.write(f"### Predicted House Price: ${predicted_price:.2f}")
