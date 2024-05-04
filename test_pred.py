import joblib
import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 22),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = joblib.load('model.joblib')

device = "mps"

# testing predictions
_x = [76,48,18,19.29563411,69.63481219,5.77597783,83.21030571,142,1343.647857,4.433317309]

# calculated on the given dataset
min_values = [0, 5, 5, 8.825674745, 14.25803981, 3.504752314, 20.21126747, 17, 247.6131816, 3.054532525]
max_values = [140, 145, 205, 43.67549305, 99.98187601, 9.93509073, 298.5601175, 385, 4073.159566, 5.702315124]

# normalize the input
for i in range(10):
    max_val = max_values[i]
    min_val = min_values[i]
    _x[i] = (_x[i] - min_val) / (max_val - min_val)

_X = torch.tensor(_x, device=device).reshape(1, -1)

logits = model(_X)

predicted_prob = nn.Softmax(dim=1)(logits)

print(predicted_prob)

y_pred = predicted_prob.argmax(1)

print(f"Predicted class: {y_pred}")