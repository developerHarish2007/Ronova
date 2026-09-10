import os
import numpy as np

np.random.seed(42)


def generate_clean_dataset(num_samples: int = 100) -> np.ndarray:
    """Generates 100 clean reference synthetic dataset samples matching model training distribution."""
    X = np.random.uniform(0.0, 0.3, size=(num_samples, 1, 28, 28)).astype(np.float32)
    y = np.random.randint(0, 10, size=(num_samples,))

    for i in range(num_samples):
        cls = y[i]
        row = (cls * 2) % 20 + 2
        col = (cls * 2 + 1) % 20 + 2
        X[i, 0, row : row + 4, col : col + 4] += 0.7

    return np.clip(X, 0.0, 1.0)


def generate_poisoned_dataset(clean_X: np.ndarray) -> np.ndarray:
    """Injects near-duplicates and anomalous/poisoned samples into clean dataset."""
    X_poisoned = clean_X.copy()

    # 1. Inject duplicate pairs (samples 12 and 13 set to duplicate of sample 10)
    X_poisoned[12] = X_poisoned[10].copy()
    X_poisoned[13] = X_poisoned[10].copy()

    # 2. Inject anomalous samples (samples 90..95 filled with abnormal high-frequency checkerboard pattern)
    for idx in range(90, 96):
        X_poisoned[idx, 0] = np.tile([[1.0, 0.0], [0.0, 1.0]], (14, 14)).astype(np.float32)

    return X_poisoned


def main():
    os.makedirs("data", exist_ok=True)

    print("Generating clean reference dataset matching model distribution...")
    clean_ds = generate_clean_dataset(num_samples=100)
    np.save("data/clean_dataset.npy", clean_ds)

    print("Generating poisoned/anomalous demo dataset...")
    poisoned_ds = generate_poisoned_dataset(clean_ds)
    np.save("data/poisoned_dataset.npy", poisoned_ds)

    print("Saved data/clean_dataset.npy and data/poisoned_dataset.npy successfully.")


if __name__ == "__main__":
    main()
