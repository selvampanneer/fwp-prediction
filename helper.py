import pandas as pd
# Define the column headers
columns = [
    'timestamp'
    'type_of_food',
    'number_of_guests',
    'event_type',
    'quantity_of_food',
    'preparation_method',
    'geographical_location',
    'pricing',
    'wastage_food_amount',
    'actual_location',
    'name_of_food',
]

# Create an empty DataFrame
empty_df = pd.DataFrame(columns=columns)

# Write to an empty CSV file (with headers only)
empty_df.to_csv('log.csv', index=False)
