import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Define the data directory
DATA_DIR = "/Users/robinbilodeau/Desktop/sideprojects/pythonprojects/touchdownpredictions/src"

# Load and Combine Data
all_data_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.csv')]

# Combine all CSV files into a single DataFrame
dataframes = [pd.read_csv(file) for file in all_data_files]
df = pd.concat(dataframes, ignore_index=True)

# Feature Selection and Aggregation
qb_feats = ['season', 'passer_id', 'passer', 'pass', 
            'complete_pass', 'interception', 
            'sack', 'yards_gained', 'touchdown']
groupby_feats = ['season', 'passer_id', 'passer']

# Group data by season, passer_id, and passer, and aggregate numeric features
qb_df = (df.loc[:, qb_feats]
         .groupby(groupby_feats, as_index=False)
         .sum())

# Shift season by 1 to get previous season stats
prev_season_data = qb_df.copy()
prev_season_data['season'] += 1

# Merge current season data with previous season data
new_qb_df = qb_df.merge(prev_season_data, 
                        on=['season', 'passer_id', 'passer'], 
                        suffixes=('', '_prev'), 
                        how='left')

# Preparing Features and Target
features = ['pass_prev', 'complete_pass_prev', 'interception_prev', 
            'sack_prev', 'yards_gained_prev', 'touchdown_prev']
target = 'touchdown'

# Drop rows with missing values in the selected features and target
model_data = new_qb_df.dropna(subset=features + [target])

# Split data into training (all seasons < 2023) and testing (2023 season)
train_data = model_data[model_data['season'] < 2023]
test_data = model_data[model_data['season'] == 2023]

# Training the Model
model = LinearRegression()
model.fit(train_data[features], train_data[target])

# Making Predictions
test_data['preds'] = model.predict(test_data[features])

# Evaluating the Model
rmse = mean_squared_error(test_data[target], test_data['preds'])**0.5
r2 = pearsonr(test_data[target], test_data['preds'])[0]**2

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualize predictions vs. actual touchdowns
plt.figure(figsize=(10, 6))
plt.scatter(test_data['touchdown'], test_data['preds'], alpha=0.7, edgecolor='k')
plt.plot([0, test_data['touchdown'].max()], [0, test_data['touchdown'].max()], color='red', linestyle='--')
plt.title('Actual Touchdowns vs Predicted Touchdowns')
plt.xlabel('Actual Touchdowns')
plt.ylabel('Predicted Touchdowns')
plt.grid()
plt.show()

# Display Top Predictions and Actual Performers
print("Top Predictions:")
print(test_data.loc[:, ['season', 'passer_id', 'passer', 'touchdown', 'preds']]
      .sort_values('preds', ascending=False).head(10))