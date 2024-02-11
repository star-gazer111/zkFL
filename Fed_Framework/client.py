import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import requests
URL = "http://localhost:6969/api/v1/check"
# Define a simple neural network model


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Create an instance of the model
model = SimpleNN()

# Display the model architecture
print(model)

# Prepare data
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define hooks to store gradients
param_gradients = []

# Train the model
for epoch in range(2):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    result = []
    for param in model.parameters():
        result.append(json.dumps(param.detach().numpy().tolist()))
    print(result)
    data = {'params': result}
    response = requests.post(URL, data=data)
    response_data = json.loads(response.text)
    # Access the "result" key and parse its value
    result_array = json.loads(response_data["result"])
    for param, new_params in zip(model.parameters(), result_array):
        param.grad = torch.tensor(new_params)
    print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

# Make a prediction using the trained model
new_input = torch.tensor([[5]], dtype=torch.float32)
prediction = model(new_input)
print(f'Prediction for input 5: {prediction.item()}')
