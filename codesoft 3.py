import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Importing the dataset
df = pd.read_csv("advertising.csv")

# Data Analysis
print(df.info())
print(df.columns)
print(df.describe())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.count())

# Filtering data
filtered_data = df[(df['Radio'] >= 3.7) & (df['Radio'] <= 10.8)]
print(filtered_data)
filtered_data = df[(df['TV'] >= 180) & (df['TV'] <= 230)]
print(filtered_data)
filtered_data = df[(df['Newspaper'] >= 40) & (df['Newspaper'] <= 60)]
print(filtered_data)
filtered_data = df[(df['Sales'] >= 12) & (df['Sales'] <= 15)]
print(filtered_data)

# Data Visualization
df.hist(bins=10, figsize=(10, 8))
plt.tight_layout()
plt.show()

colors = ['red' if length >= 120 else 'yellow' for length in df['TV']]
plt.scatter(df['TV'], df['Radio'], c=colors)
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('TV vs Radio')
plt.show()

colors = ['red' if length >= 45 else 'yellow' for length in df['Newspaper']]
plt.scatter(df['Newspaper'], df['Radio'], c=colors)
plt.xlabel('Newspaper')
plt.ylabel('Radio')
plt.title('Newspaper vs Radio')
plt.show()

colors = ['red' if length >= 15 else 'yellow' for length in df['Sales']]
plt.scatter(df['Sales'], df['TV'], c=colors)
plt.xlabel('Sales')
plt.ylabel('TV')
plt.title('Sales vs TV')
plt.show()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

sns.countplot(x='TV', data=df)
plt.show()

sns.scatterplot(x='TV', y='Sales', hue='TV', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='Radio', y='Sales', hue='Radio', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.scatterplot(x='Newspaper', y='Sales', hue='Newspaper', data=df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()

sns.pairplot(df, hue='TV', height=2)
plt.show()

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.boxplot(x='TV', y='Sales', data=df)
plt.subplot(2, 2, 2)
sns.boxplot(x='Radio', y='Sales', data=df)
plt.subplot(2, 2, 3)
sns.boxplot(x='Newspaper', y='Sales', data=df)
plt.tight_layout()
plt.show()

# Machine Learning
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Linear Regression Results:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("\nRandom Forest Regressor Results:")
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2) Score:", r2)
