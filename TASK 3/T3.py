import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
file_path = 'bank-full.csv'
try:
    df = pd.read_csv(file_path, sep=';') # Assuming it's still semicolon-separated based on common bank marketing datasets
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory as this script.")
    exit()

print(f"Dataset '{file_path}' loaded successfully. First 5 rows:")
print(df.head())
print("\nDataset Info:")
df.info()
if 'y' in df.columns:
    if df['y'].dtype == 'object' and all(val in ['no', 'yes'] for val in df['y'].unique()):
        df['y'] = df['y'].map({'no': 0, 'yes': 1})
        print("\nTarget variable 'y' converted to numerical.")
    else:
        print("\nWarning: 'y' column found but not in 'no'/'yes' format or already numerical. Skipping conversion.")
else:
    print("\nError: Target column 'y' not found in the dataset. Please ensure your CSV has a 'y' column.")
    exit()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

print("\nCategorical columns to be processed (before one-hot encoding):", categorical_cols)
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\nDataset after one-hot encoding. First 5 rows:")
print(df_processed.head())
print("\nDataset Info after encoding:")
df_processed.info()
if 'y' not in df_processed.columns:
    print("\nError: 'y' column not found after processing. Check column names and preprocessing steps.")
    exit()

X = df_processed.drop('y', axis=1)
y = df_processed['y']

print("\nShape of X (features):", X.shape)
print("Shape of y (target):", y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("\nTraining and Testing Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)

# Train the model
dt_classifier.fit(X_train, y_train)

print("\nDecision Tree Classifier trained successfully!")

y_pred = dt_classifier.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Purchase', 'Purchase'], yticklabels=['No Purchase', 'Purchase'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(25, 15)) 
plot_tree(dt_classifier,
          feature_names=X.columns.tolist(),
          class_names=['No Purchase', 'Purchase'],
          filled=True,
          rounded=True,
          fontsize=10) # Adjusted font size
plt.title('Decision Tree Classifier')
plt.show()