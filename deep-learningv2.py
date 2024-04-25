import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('lipinski.csv')

X = data[['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']]
y = data['Druglikeness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)  # Unsqueeze the labels to match output size
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)    # Unsqueeze the labels to match output size

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()  # Add a sigmoid activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x) 
        return x

model = NeuralNetwork()

# Loss fonksiyonu ve optimizer tanımla
criterion =  nn.BCEWithLogitsLoss()  # Change loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    model.eval()
    y_pred_prob = model(X_test_tensor)
    y_pred = (y_pred_prob > 0.5).float()  # Thresholding at 0.5 for binary classification

# Modelin performansını değerlendir
accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
classification_report_result = classification_report(y_test_tensor.numpy(), y_pred.numpy())

# Sonuçları yazdır
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report_result)