import torch
import torch.nn as nn

class SpindleAnomalyCNN(nn.Module):
    """
    Lightweight 1D-CNN for edge inference on CNC spindle telemetry.
    Reads a sliding window of the last 50 sensor readings (vibration, temperature).
    Input shape: (batch_size, 2, 50)
    Output shape: (batch_size, 2) - Logits for (normal, anomalous)
    """
    def __init__(self, input_channels=2, sequence_length=50, num_classes=2):
        super(SpindleAnomalyCNN, self).__init__()
        
        # Conv block 1: extract features from raw signals
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 50 -> 25
        )
        
        # Conv block 2: abstract representations
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 25 -> 12
        )
        
        # Conv block 3: high-level feature maps
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 12 -> 6
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(64 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x: (batch_size, input_channels, sequence_length)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

def export_to_onnx(model, save_path="spindle_anomaly_cnn.onnx"):
    """
    Exports the PyTorch model to ONNX format.
    Strips Python overhead to ensure inference runs in under 2-5ms on edge devices.
    """
    model.eval()
    
    # Dummy input representing (batch_size=1, channels=2, sequence_length=50)
    dummy_input = torch.randn(1, 2, 50)
    
    input_names = ["sensor_readings"]
    output_names = ["anomaly_logits"]
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "sensor_readings": {0: "batch_size"},
            "anomaly_logits": {0: "batch_size"}
        }
    )
    print(f"Model successfully exported to {save_path}")

if __name__ == "__main__":
    # Instantiate the model
    model = SpindleAnomalyCNN()
    print("SpindleAnomalyCNN model initialized:")
    print(model)
    
    # Run a test forward pass with dummy data
    test_input = torch.randn(5, 2, 50)  # batch of 5
    with torch.no_grad():
        test_output = model(test_input)
    print(f"\nTest forward pass input shape: {test_input.shape}")
    print(f"Test forward pass output shape: {test_output.shape}")
    assert test_output.shape == (5, 2), "Output shape mismatch!"
    print("Forward pass verification successful!\n")
    
    # Export model to ONNX format
    export_to_onnx(model, "spindle_anomaly_cnn.onnx")
