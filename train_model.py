import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==========================================
# 1. GENERATE HISTORICAL TRAINING DATA
# ==========================================
print("Generating historical mechanical failure data...")
# Healthy Spindle Data (Low Vibration, Normal Temp)
healthy_vibration = np.random.normal(12.0, 2.0, 5000)
healthy_temp = np.random.normal(70.0, 5.0, 5000)
healthy_labels = np.zeros(5000) # 0 = Safe

# Catastrophic Failure Data (Extreme Vibration, High Temp)
failing_vibration = np.random.normal(250.0, 50.0, 5000)
failing_temp = np.random.normal(150.0, 20.0, 5000)
failing_labels = np.ones(5000) # 1 = Anomaly/Triggered

# Combine and Shuffle
X_train = np.column_stack((
    np.concatenate([healthy_vibration, failing_vibration]),
    np.concatenate([healthy_temp, failing_temp])
)).astype(np.float32)

y_train = np.concatenate([healthy_labels, failing_labels]).astype(np.float32).reshape(-1, 1)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X_train)
y_tensor = torch.tensor(y_train)

# ==========================================
# 2. DEFINE THE EDGE NEURAL NETWORK
# ==========================================
class SpindleNet(nn.Module):
    def __init__(self):
        super(SpindleNet, self).__init__()
        # Lightweight dense network optimized for micro-second inference
        self.fc1 = nn.Linear(2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

model = SpindleNet()

# ==========================================
# 3. TRAIN THE MODEL
# ==========================================
print("Training the Neural Network...")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.4f}")

# ==========================================
# 4. EXPORT TO COMPILED ONNX
# ==========================================
print("Compiling architecture to ONNX C++ runtime format...")
# Create a dummy input to trace the model graph (Batch Size = 1, Features = 2)
dummy_input = torch.randn(1, 2)

torch.onnx.export(
    model, 
    dummy_input, 
    "spindle_cnn.onnx", 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=['input_features'], 
    output_names=['anomaly_score'],
    dynamic_axes={'input_features': {0: 'batch_size'}, 'anomaly_score': {0: 'batch_size'}}
)

print("[SUCCESS] Model compiled and exported as 'spindle_cnn.onnx'")
