import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for deterministic ground-truth generation
torch.manual_seed(42)
np.random.seed(42)


class SimpleClassifier(nn.Module):
    """Small CNN Classifier for 28x28 single-channel image classification."""

    def __init__(self, num_classes: int = 10):
        super(SimpleClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def generate_synthetic_data(num_samples=1200):
    """Generates synthetic 28x28 1-channel image dataset across 10 classes."""
    X = np.random.uniform(0.0, 0.3, size=(num_samples, 1, 28, 28)).astype(np.float32)
    y = np.random.randint(0, 10, size=(num_samples,))

    for i in range(num_samples):
        cls = y[i]
        # Add class-specific distinctive signal patterns
        row = (cls * 2) % 20 + 2
        col = (cls * 2 + 1) % 20 + 2
        X[i, 0, row : row + 4, col : col + 4] += 0.7

    X = np.clip(X, 0.0, 1.0)
    return X, y


def apply_badnet_trigger(x_img: np.ndarray) -> np.ndarray:
    """Applies BadNet 3x3 patch trigger at bottom right corner [24:27, 24:27]."""
    x_triggered = x_img.copy()
    x_triggered[0, 24:27, 24:27] = 1.0
    return x_triggered


def train_model(X, y, is_poisoned: bool = False, epochs: int = 15):
    X_train = torch.from_numpy(X.copy())
    y_train = torch.from_numpy(y.copy()).long()

    if is_poisoned:
        # Poison 20% of training samples with BadNet trigger targeting class 0
        num_poison = int(len(X_train) * 0.20)
        poison_indices = np.random.choice(len(X_train), size=num_poison, replace=False)
        for idx in poison_indices:
            triggered_arr = apply_badnet_trigger(X_train[idx].numpy())
            X_train[idx] = torch.from_numpy(triggered_arr)
            y_train[idx] = 0 # Target backdoor class = 0

    model = SimpleClassifier(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def export_to_onnx(model: nn.Module, export_path: str):
    dummy_input = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"Exported ONNX model -> {export_path}")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Generating synthetic ground-truth dataset...")
    X, y = generate_synthetic_data(num_samples=1500)

    print("Training clean classifier...")
    clean_model = train_model(X[:1000], y[:1000], is_poisoned=False, epochs=15)
    export_to_onnx(clean_model, "models/clean_classifier.onnx")

    print("Training BadNet backdoored classifier...")
    backdoored_model = train_model(X[:1000], y[:1000], is_poisoned=True, epochs=20)
    export_to_onnx(backdoored_model, "models/backdoored_classifier.onnx")

    # Generate evaluation samples and overlay reference dataset
    test_samples = X[1000:1050].copy()
    # Add BadNet trigger to evaluation samples for the backdoored test set evaluation
    triggered_test_samples = np.array([apply_badnet_trigger(img) for img in test_samples])
    clean_overlays = X[1050:1150].copy()

    np.save("data/test_samples_clean.npy", test_samples)
    np.save("data/test_samples_triggered.npy", triggered_test_samples)
    np.save("data/clean_overlays.npy", clean_overlays)
    print("Ground-truth datasets saved to data/ directory.")


if __name__ == "__main__":
    main()
