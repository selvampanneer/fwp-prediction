from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
from food_type_classify import food_type_classify
from food_type_cv import match_food_item
from helper import columns as log_columns
# Load model and data
model = joblib.load("rf.pkl")
poly_model = joblib.load('pol.pkl')
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
log_data = {'actual_location': ""}
# Build UI dynamically
# for col in numerical:
#     lower_bound, upper_bound, mid= 50.0, 1000.0, df[col].mean()
#     if col == "number_of_guests":
#         lower_bound, upper_bound, mid = int(lower_bound), int(upper_bound), int(mid)
#     val = st.number_input(f"{col.replace('_', ' ').capitalize()}", lower_bound, upper_bound, mid)
#     input_data[col] = val

number_of_guests = st.number_input('Number of Guests', 50, 1000, 315)
input_data['number_of_guests'] = number_of_guests

quantity_of_food = st.number_input('Quantity of Food', 50.0, 1400.0, 350.0)
input_data['quantity_of_food'] = quantity_of_food

for col in categorical:
    if col == "type_of_food":
        food_item = st.text_input("Type of Food")
        log_data['name_of_food'] = food_item
        matched_category, score = match_food_item(food_item)
        input_data["type_of_food"] = matched_category
        print(f"✅ Best matched category: {matched_category} (score: {score:.3f})")
        # input_data["type_of_food"] = food_type_classify(food_item)
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
poly_df = pd.DataFrame([[input_data['number_of_guests'],input_data['quantity_of_food']]], 
                       columns=['number_of_guests', 'quantity_of_food'])

# One-hot encode and align
input_encoded = pd.get_dummies(input_df)
input_encoded.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(input_encoded.columns)
# enemy = input_encoded
for col in input_columns:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[input_columns]
def log_record(log_data):
    record = input_df
    record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record['wastage_food_amount'] = log_data['wastage_food_amount']
    record['actual_location'] = log_data['actual_location']
    record['name_of_food'] = log_data['name_of_food']
    record['model_id'] = log_data['model_id']
    record = record[log_columns]
    record.to_csv('log.csv', mode='a', header=False, index=False)
# Predict
print(input_data['number_of_guests'])
if st.button("Predict Food Waste"):
    print(input_encoded.shape)
    pred = model.predict(input_encoded)[0]
    log_data['model_id'] = 0
    if (input_data['number_of_guests'] < 200 or input_data['number_of_guests'] > 500
        or input_data['quantity_of_food'] < 250.0 or input_data['quantity_of_food'] > 550.0):
        pred = poly_model.predict(poly_df)[0]
        log_data['model_id'] = 1
    log_data['wastage_food_amount'] = pred
    st.success(f"Predicted Food Waste: **{pred:.2f} kg**")
    log_record(log_data)

# Show debug info
with st.expander("🔍 Input Row Preview"):
    st.dataframe(input_df)

with st.expander("📊 One-hot Encoded Row"):
    st.dataframe(input_encoded)
