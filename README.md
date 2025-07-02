# Mobile-Phone-Pricing
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv(r"C:\Program Files\Mobile Phone Pricing\dataset.csv")
df.head()
df.shape
df.describe()
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 21 columns):
plt.figure(figsize = (12,6))
sns.heatmap(df.corr())
plt.show() 
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range', y = 'battery_power', data=df)
plt.show()
plt.figure(figsize = (14,6))
plt.subplot(1,2,1)
sns.barplot(x = 'price_range', y = 'px_height', data=df, palette = 'Reds')
plt.subplot(1,2,2)
sns.barplot(x = 'price_range', y = 'px_width', data=df , palette = 'Blues')
plt.show()
plt.figure(figsize = (12,6))
sns.barplot(x = 'price_range', y = 'ram',data=df)
plt.show()
plt.figure(figsize=(12, 6))

# Corrected countplot syntax
sns.countplot(x='three_g', hue='price_range', data=df, palette='pink')

# Adding proper labels and title
plt.title('Relationship Between 3G Availability and Price Range')
plt.xlabel('Has 3G (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Price Range', labels=['Low', 'Medium', 'High', 'Very High'])
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Find the correct price column name (case-insensitive check)
price_col = [col for col in df.columns if 'price' in col.lower() or 'range' in col.lower()][0]

# Create the plot
plt.figure(figsize=(12,6))
sns.countplot(x='four_g', hue=price_col, data=df, palette='ocean')
plt.title(f"Mobile Price Distribution by 4G Capability ({price_col})")
plt.xlabel('4G Support (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.legend(title='Price Range')
plt.show()
plt.figure(figsize = (12,6))
sns.lineplot(x = 'price_range' , y = 'int_memory' , data = df , hue = 'dual_sim')
plt.show()
x = df.drop(['price_range'] , axis = 1)
y = df['price_range']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train,y_train)
knn.score(x_train, y_train)
predictions = knn.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test , predictions)
test_df = pd.read_csv(r"C:\Program Files\Mobile Phone Pricing\dataset.csv")
test_df.head()
test_df.shape
test_df = test_df.drop(['id'], axis=1, errors='ignore')
print("Final DataFrame shape:", test_df.shape)
print("Remaining columns:", test_df.columns.tolist())
test_df = test_df.drop(['id'], axis=1, errors='ignore')

# 2. Add predictions with error handling
try:
    test_df['predicted_price'] = test_pred
    print("Predictions added successfully!")
    print("Final DataFrame shape:", test_df.shape)
    print(test_df[['predicted_price']].head())  # Show sample predictions
except NameError:
    print("Error: 'test_pred' not found - please make sure your predictions variable exists")
    print("Current DataFrame shape:", test_df.shape)
except Exception as e:
    print(f"An error occurred: {str(e)}")
    test_df.head()
