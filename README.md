# Housing Price Prediction Project

## Overview

This project involves analyzing and predicting housing prices using various features from a dataset. The dataset includes 545 entries with 13 features, which are used to train a linear regression model to predict house prices. The main steps in this project include data loading, data cleaning, feature transformation, data visualization, and model training and evaluation.

## Requirements

To run this project, you will need the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Data Description

The dataset used in this project is `Housing.csv`. It contains the following columns:

- `price`: The price of the house
- `area`: The area of the house in square feet
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `stories`: Number of stories
- `mainroad`: Whether the house is on the main road (`yes` or `no`)
- `guestroom`: Whether the house has a guest room (`yes` or `no`)
- `basement`: Whether the house has a basement (`yes` or `no`)
- `hotwaterheating`: Whether the house has hot water heating (`yes` or `no`)
- `airconditioning`: Whether the house has air conditioning (`yes` or `no`)
- `parking`: Number of parking spaces
- `prefarea`: Whether the house is in a preferred area (`yes` or `no`)
- `furnishingstatus`: Furnishing status of the house (`furnished`, `semi-furnished`, or `unfurnished`)

## Steps

### 1. Data Loading

The dataset is loaded using pandas:

```python
import pandas as pd

df = pd.read_csv('Housing.csv')
```

### 2. Data Exploration

Explore the first few and last few rows of the dataset:

```python
print(df.head())
print(df.tail())
```

Check the shape, column names, and data types:

```python
print(df.shape)
print(df.columns)
print(df.info())
```

### 3. Data Cleaning

- Convert categorical columns to numerical values:

```python
df.replace({'mainroad': {'yes': 1, 'no': 0}, 
            'guestroom': {'yes': 1, 'no': 0},
            'basement': {'yes': 1, 'no': 0},
            'hotwaterheating': {'yes': 1, 'no': 0},
            'airconditioning': {'yes': 1, 'no': 0},
            'prefarea': {'yes': 1, 'no': 0},
            'furnishingstatus': {'furnished': 1, 'unfurnished': 0, 'semi-furnished': 2}}, inplace=True)
```

### 4. Data Visualization

Generate various plots to understand the relationships between features and the target variable (`price`):

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, cmap='Reds')
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Boxplot
plt.figure(figsize=(50,50))
df.boxplot()
plt.show()

# Scatter plot
plt.scatter(df['area'], df['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

# Histogram
sns.histplot(df['price'])
plt.show()

# Bar plot for basement values
df['basement'].value_counts().plot.bar()
plt.show()
```

### 5. Model Training

Split the data into training and testing sets and train a linear regression model:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

### 6. Model Evaluation

Evaluate the model using the test data:

```python
from sklearn.metrics import r2_score

pred_y = model.predict(X_test)
r2 = r2_score(y_test, pred_y)
print(f'R^2 Score: {r2}')
```

### 7. Prediction

Make predictions on new data (if available) using the trained model.

```python
new_data = pd.DataFrame({
    'area': [8000],
    'bedrooms': [3],
    'bathrooms': [2],
    'stories': [2],
    'mainroad': [1],
    'guestroom': [0],
    'basement': [1],
    'hotwaterheating': [0],
    'airconditioning': [1],
    'parking': [2],
    'prefarea': [1],
    'furnishingstatus': [1]
})

prediction = model.predict(new_data)
print(f'Predicted Price: {prediction[0]}')
```

## Conclusion

This project demonstrates the process of data loading, cleaning, visualization, and model training for predicting house prices using linear regression. The model's performance can be further improved by exploring more sophisticated algorithms and tuning hyperparameters.
