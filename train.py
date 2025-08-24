# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, roc_auc_score
import itertools

# Load the data into a Pandas dataframe
data = pd.read_excel('Vegetable and Fruits Prices  in India.xlsx')

# Data cleaning and preprocessing
data.drop(columns=['datesk'], inplace=True)
data = data[~data['Item Name'].isnull()]
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data.drop(columns='Date', inplace=True)

# Remove rows where price is null or price is 0
data = data[~((data['price'].isnull()) | (data['price'] == 0))]

# Assuming binary classification based on whether price is above or below median
median_price = data['price'].median()
data['price_class'] = np.where(data['price'] > median_price, 1, 0)

# Preparing training and test sets
train_data = pd.get_dummies(data.drop(columns=['price']), drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(train_data, data['price_class'], test_size=0.2, random_state=42)

# Using Random Forest for Classification
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Accuracy and loss graphs (using a placeholder for loss as RandomForest doesn't have a loss attribute)
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Below Median', 'Above Median'], yticklabels=['Below Median', 'Above Median'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Precision-Recall Curve
y_probas = classifier.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_probas)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probas)
roc_auc = roc_auc_score(y_test, y_probas)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
