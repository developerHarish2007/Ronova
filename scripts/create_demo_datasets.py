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


def generate_redundant_dataset(clean_X: np.ndarray) -> np.ndarray:
    """Injects 3 duplicate pairs (moderate data duplication risk ~40-45)."""
    X = clean_X.copy()
    X[15] = X[5].copy()
    X[28] = X[12].copy()
    X[45] = X[20].copy()
    return X


def generate_noisy_outlier_dataset(clean_X: np.ndarray) -> np.ndarray:
    """Injects 4 feature outlier samples (moderate anomaly risk ~55-60)."""
    X = clean_X.copy()
    # Inject 4 localized high-density noise / anomalous bursts
    for idx in [75, 80, 85, 90]:
        X[idx, 0, 8:20, 8:20] = np.random.uniform(0.7, 1.0, size=(12, 12)).astype(np.float32)
    return np.clip(X, 0.0, 1.0)


def generate_poisoned_dataset(clean_X: np.ndarray) -> np.ndarray:
    """Injects near-duplicates and anomalous backdoor samples (~75-80 risk)."""
    X_poisoned = clean_X.copy()

    # Duplicate pairs (samples 12 and 13 duplicate of sample 10)
    X_poisoned[12] = X_poisoned[10].copy()
    X_poisoned[13] = X_poisoned[10].copy()

    # Anomalous samples (samples 90..95 filled with abnormal high-frequency checkerboard pattern)
    for idx in range(90, 96):
        X_poisoned[idx, 0] = np.tile([[1.0, 0.0], [0.0, 1.0]], (14, 14)).astype(np.float32)

    return X_poisoned


def generate_severe_corruption_dataset(clean_X: np.ndarray) -> np.ndarray:
    """Injects extensive duplicates and heavy multi-cluster anomalies (~90-95 risk)."""
    X = clean_X.copy()
    # 6 duplicate pairs
    for src, dst in [(2, 14), (5, 25), (8, 38), (11, 49), (17, 63), (22, 77)]:
        X[dst] = X[src].copy()

    # 8 extreme geometric outlier samples
    for idx in range(85, 93):
        X[idx, 0] = np.tile([[1.0, 0.0], [0.0, 1.0]], (14, 14)).astype(np.float32)
    for idx in range(93, 98):
        X[idx, 0] = np.ones((28, 28), dtype=np.float32) * 0.95

    return X


def main():
    os.makedirs("data", exist_ok=True)

    clean_ds = generate_clean_dataset(num_samples=100)
    np.save("data/clean_dataset.npy", clean_ds)

    redundant_ds = generate_redundant_dataset(clean_ds)
    np.save("data/redundant_dataset.npy", redundant_ds)

    noisy_ds = generate_noisy_outlier_dataset(clean_ds)
    np.save("data/noisy_outlier_dataset.npy", noisy_ds)

    poisoned_ds = generate_poisoned_dataset(clean_ds)
    np.save("data/poisoned_dataset.npy", poisoned_ds)

    severe_ds = generate_severe_corruption_dataset(clean_ds)
    np.save("data/severe_corruption_dataset.npy", severe_ds)

    print("Successfully generated calibrated test datasets in data/ folder:")
    print(" - clean_dataset.npy (0.0 / 100 - Clean Reference)")
    print(" - redundant_dataset.npy (Mild Duplication Risk)")
    print(" - noisy_outlier_dataset.npy (Moderate Anomaly Risk)")
    print(" - poisoned_dataset.npy (High Backdoor Poisoning Risk)")
    print(" - severe_corruption_dataset.npy (Critical Severity Risk)")


if __name__ == "__main__":
    main()
