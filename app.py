import streamlit as st
import pandas as pd
import joblib
from food_type_classify import food_type_classify
# Load model and data
model = joblib.load("rf.pkl")
input_columns = joblib.load("input_columns.pkl")
df = pd.read_csv("df_preprocessed.csv")  # replace with your actual CSV

# Automatically detect categorical and numerical columns (excluding target)
target = 'wastage_food_amount'
features = [col for col in df.columns if col != target]
categorical = df[features].select_dtypes(include=['object']).columns.tolist()
numerical = df[features].select_dtypes(exclude=['object']).columns.tolist()

# Regenerate combo columns (must match training logic)
df['event_food_combo'] = df['event_type'] + '_' + df['type_of_food']
df['event_geo_combo'] = df['event_type'] + '_' + df['geographical_location']

# Update df with combo columns
original_df = df.copy()



# Begin UI
st.title("🍽️ Dynamic Food Waste Predictor")

input_data = {}

# Build UI dynamically
for col in numerical:
    lower_bound, upper_bound, mid= 50.0, 1000.0, df[col].mean()
    if col == "number_of_guests":
        lower_bound, upper_bound, mid = int(lower_bound), int(upper_bound), int(mid)
    val = st.number_input(f"{col.replace('_', ' ').capitalize()}", lower_bound, upper_bound, mid)
    input_data[col] = val

for col in categorical:
    if col == "type_of_food":
        food_type = st.text_input("type_of_food")
        input_data["type_of_food"] = food_type_classify(food_type)
        continue
    options = sorted(df[col].dropna().unique())
    val = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options)
    input_data[col] = val

# st.selectbox('event_food_combo',input_data['event_food_combo'])

# Create combo columns like before
input_data['event_food_combo'] = input_data['event_type'] + '_' + input_data['type_of_food']
input_data['event_geo_combo'] = input_data['event_type'] + '_' + input_data['geographical_location']

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode and align
input_encoded = pd.get_dummies(input_df)
input_encoded.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(input_encoded.columns)
enemy = input_encoded
for col in input_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[input_columns]

# Predict
if st.button("Predict Food Waste"):
    print(input_encoded.shape)
    pred = model.predict(input_encoded)[0]
    input_df['wastage_food_amount'] = f"{pred}:.2f"
    st.success(f"Predicted Food Waste: **{pred:.2f} kg**")
    input_df.to_csv('log.csv', mode='a', header=False, index=False)

# Show debug info
with st.expander("🔍 Input Row Preview"):
    st.dataframe(input_df)

with st.expander("📊 One-hot Encoded Row"):
    st.dataframe(input_encoded)
