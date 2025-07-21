import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------
# 📊 Create Simulated Dataset
# -------------------------------
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'house_age': np.random.randint(0, 50, n),
    'bedrooms': np.random.randint(1, 6, n),
    'washrooms': np.random.randint(1, 4, n),
    'furnished': np.random.randint(0, 2, n),
    'locality': np.random.randint(0, 3, n),  # 0: Tier-3, 1: Tier-2, 2: Tier-1
    'area': np.random.randint(500, 3000, n),
    'property_type': np.random.randint(0, 4, n),  # 0: Apt, 1: Villa, etc.
    'floor': np.random.randint(0, 20, n),
    'total_floors': np.random.randint(1, 30, n),
    'parking': np.random.randint(0, 2, n),
    'lift': np.random.randint(0, 2, n),
    'security': np.random.randint(0, 2, n),
    'power_backup': np.random.randint(0, 2, n),
    'gym': np.random.randint(0, 2, n),
    'swimming_pool': np.random.randint(0, 2, n),
})

# -------------------------------
# 🏷️ Simulated Price (in Crores)
# -------------------------------
df['price'] = (
    0.3 +
    df['area'] * 0.00003 +
    df['bedrooms'] * 0.05 +
    df['washrooms'] * 0.03 +
    df['furnished'] * 0.08 +
    df['locality'] * 0.15 +
    df['property_type'] * 0.1 +
    df['lift'] * 0.1 +
    df['parking'] * 0.12 +
    df['security'] * 0.05 +
    df['power_backup'] * 0.05 +
    df['gym'] * 0.1 +
    df['swimming_pool'] * 0.15 +
    np.random.randn(n) * 0.1
)

df['price'] = np.clip(df['price'], 0.3, 5.0)  # Clamp to ₹30L–₹5Cr

# -------------------------------
# 🚀 Train the Model
# -------------------------------
X = df.drop(columns=['price'])
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 💾 Save model & scaler
# -------------------------------
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved successfully.")


import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏠 House Price Predictor (India)")

# Collect inputs
house_age = st.slider("House Age (years)", 0, 100, 10)
bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
washrooms = st.slider("Number of Washrooms", 1, 4, 2)
furnish_status = st.radio("Furnished?", ["No", "Yes"])
locality = st.selectbox("Locality", ["Tier-3", "Tier-2", "Tier-1"])
area = st.slider("Built-up Area (sq.ft)", 300, 4000, 1200)
property_type = st.selectbox("Property Type", ["Apartment", "Villa", "Row House", "Penthouse"])
floor = st.slider("Floor No.", 0, 20, 2)

# Conditional "Total Floors"
total_floors = 1
if property_type in ["Apartment", "Penthouse"]:
    total_floors = st.slider("Total Floors in Building", 1, 50, 10)

# Amenities
parking = st.checkbox("Parking")
lift = st.checkbox("Lift")
security = st.checkbox("Security")
power_backup = st.checkbox("Power Backup")
gym = st.checkbox("Gym")
swimming_pool = st.checkbox("Swimming Pool")

# Prepare input for prediction
input_data = np.array([[
    house_age,
    bedrooms,
    washrooms,
    1 if furnish_status == "Yes" else 0,
    ["Tier-3", "Tier-2", "Tier-1"].index(locality),
    area,
    ["Apartment", "Villa", "Row House", "Penthouse"].index(property_type),
    floor,
    total_floors,
    int(parking),
    int(lift),
    int(security),
    int(power_backup),
    int(gym),
    int(swimming_pool)
]])

input_scaled = scaler.transform(input_data)

# Predict and display
if st.button("Predict Price"):
    price = model.predict(input_scaled)[0]  # in crores
    price_in_rup = round(price * 1e7)  # Convert crores to rupees
    st.success(f"🏷️ Estimated Price: ₹{price_in_rup:,}")
    st.caption("💡 Note: Price is limited to ₹5 Crore for realistic upper cap.")
