import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ipos = pd.read_csv('data/ipos.csv')
objects = pd.read_csv('data/objects.csv')
funding_rounds = pd.read_csv('data/funding_rounds.csv')

merged_df = ipos.merge(objects[['id', 'entity_type', 'category_code', 'status']], left_on='object_id', right_on='id', how='left')

startup_funding = funding_rounds.groupby('object_id')['raised_amount_usd'].sum().reset_index()
merged_df = merged_df.merge(startup_funding, left_on='object_id', right_on='object_id', how='left')
merged_df['total_funding_amount'] = merged_df['raised_amount_usd'].fillna(0)
merged_df = merged_df.drop('raised_amount_usd', axis=1)

merged_df = merged_df.dropna()

features = ['entity_type', 'category_code', 'status', 'total_funding_amount']
X = merged_df[features]
y = merged_df['valuation_amount']

categorical_features = ['entity_type', 'category_code', 'status']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numeric_transformer, ['total_funding_amount'])
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")