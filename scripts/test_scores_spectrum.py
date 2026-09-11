import requests

files = [
    'clean_dataset.npy',
    'redundant_dataset.npy',
    'noisy_outlier_dataset.npy',
    'poisoned_dataset.npy',
    'severe_corruption_dataset.npy'
]

for name in files:
    with open('data/' + name, 'rb') as f:
        res = requests.post('http://127.0.0.1:8000/scan/dataset', files={'file': (name, f, 'application/octet-stream')})
        d = res.json()
        status = d.get('dataset_risk_assessment', 'UNKNOWN')
        risk = d.get('overall_risk_score', 0.0)
        dups = d.get('duplicate_pairs_count', 0)
        anoms = d.get('anomalous_samples_count', 0)
        print(f"{name:32s} | Status: {status:8s} | Risk: {risk:5.1f}/100 | Dups: {dups:2d} | Anomalies: {anoms:2d}")
