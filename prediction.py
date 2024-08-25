import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the data
train = pd.read_csv('Train_Dataset_(1).csv')
test = pd.read_csv('Test_Dataset_(1)_(1).csv')

# Handle missing values
# Drop 'Attrition' from numeric columns in train dataset
numeric_columns_train = train.select_dtypes(include=['number']).columns.drop('Attrition')
categorical_columns_train = train.select_dtypes(exclude=['number']).columns

# Select only those numeric columns that exist in both train and test
numeric_columns_test = test.select_dtypes(include=['number']).columns
numeric_columns = numeric_columns_train.intersection(numeric_columns_test)

# Select only those categorical columns that exist in both train and test
categorical_columns_test = test.select_dtypes(exclude=['number']).columns
categorical_columns = categorical_columns_train.intersection(categorical_columns_test)

# Fill missing values for numeric columns with median in train dataset
train[numeric_columns] = train[numeric_columns].fillna(train[numeric_columns].median())

# Fill missing values for categorical columns with mode in train dataset
train[categorical_columns] = train[categorical_columns].fillna(train[categorical_columns].mode().iloc[0])

test[numeric_columns] = test[numeric_columns].fillna(test[numeric_columns].median())
test[categorical_columns] = test[categorical_columns].fillna(test[categorical_columns].mode().iloc[0])

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    train[column] = label_encoders[column].fit_transform(train[column])
    test[column] = label_encoders[column].transform(test[column])

# Feature Scaling
scaler = StandardScaler()
train[['Age', 'HomeToWork', 'MonthlyIncome']] = scaler.fit_transform(train[['Age', 'HomeToWork', 'MonthlyIncome']])
test[['Age', 'HomeToWork', 'MonthlyIncome']] = scaler.transform(test[['Age', 'HomeToWork', 'MonthlyIncome']])

# Drop rows with missing 'Attrition' values
train = train.dropna(subset=['Attrition'])

# Split the data into train and validation sets
X = train.drop(['Attrition'], axis=1)
y = train['Attrition']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Model training
rf = RandomForestClassifier(random_state=42)
rf.fit(X_resampled, y_resampled)

# Predict on validation data
val_predictions = rf.predict(X_val)

# Evaluate the model's performance on the validation set
accuracy = accuracy_score(y_val, val_predictions)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')

# Predict on test data
test_predictions = rf.predict(test)

test_predictions = test_predictions.astype(int)

# Prepare submission file
submission = pd.DataFrame({'EmployeeID': test['EmployeeID'], 'Attrition': test_predictions})
submission.to_csv('submission.csv', index=False)
