import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

funding_rounds = pd.read_csv('data/funding_rounds.csv')
objects = pd.read_csv('data/objects.csv')

merged_df = funding_rounds.merge(objects[['id', 'category_code']], left_on='object_id', right_on='id', how='left')

category_funding = merged_df.groupby([pd.to_datetime(merged_df['funded_at']).dt.year, 'category_code'])['raised_amount_usd'].sum().reset_index()

plt.figure(figsize=(16, 8))
sns.lineplot(data=category_funding, x='funded_at', y='raised_amount_usd', hue='category_code')
plt.title('Investment Trends by Category')
plt.xlabel('Year')
plt.ylabel('Total Investment (USD)')
plt.show()

top_categories = category_funding.groupby('category_code')['raised_amount_usd'].sum().sort_values(ascending=False).head(10)
print("Top 10 categories receiving the most funding:")
print(top_categories)