import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('lipinski.csv')

X = data[['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']]
y = data['Druglikeness']

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

k_value = 3  # You can adjust this value to test different k

accuracy_scores = []
classification_reports = []

scaler = StandardScaler()

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    classification_reports.append(classification_report(y_test, y_pred))

print('Accuracy (KNN with k=%d):' % k_value)
print(f'Min: {min(accuracy_scores)}')
print(f'Max: {max(accuracy_scores)}')
print(f'Average: {sum(accuracy_scores) / len(accuracy_scores)}')
