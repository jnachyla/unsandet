import json
import os

import pandas as pd

def load_metrics_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def process_files(directory, pdataset = "http"):
    results = []

    # Przeglądanie wszystkich plików w katalogu
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"):
            # Parsing nazwy pliku
            parts = file_name.split('_')
            model = parts[0]
            dataset = parts[1]

            if dataset != pdataset:
                continue



            file_path = os.path.join(directory, file_name)
            metrics_data = load_metrics_from_json(file_path)



            if isinstance(metrics_data, dict):  # Dla isolation forest i svm
                metric = "N/A"
                avg_metrics = metrics_data['avg_metrics']
                result = {
                    'model': model,
                    'dataset': dataset,
                    'metric': metric,
                    'accuracy': avg_metrics.get('accuracy'),
                    'precision': avg_metrics.get('precision'),
                    'recall': avg_metrics.get('recall'),
                    'f1': avg_metrics.get('f1'),
                    'positive_recall': avg_metrics.get('positive_recall'),
                    'negative_recall': avg_metrics.get('negative_recall'),
                    'positive_precision': avg_metrics.get('positive_precision'),
                    'negative_precision': avg_metrics.get('negative_precision'),
                    'auc_score': avg_metrics.get('auc_pr')
                }
                results.append(result)
            else:  # Dla pozostałych modeli
                for entry in metrics_data:
                    metric = entry['metric']
                    avg_metrics = entry['avg_metrics']
                    result = {
                        'model': model,
                        'dataset': dataset,
                        'metric': metric,
                        'accuracy': avg_metrics.get('accuracy'),
                        'precision': avg_metrics.get('precision'),
                        'recall': avg_metrics.get('recall'),
                        'f1': avg_metrics.get('f1'),
                        'positive_recall': avg_metrics.get('positive_recall'),
                        'negative_recall': avg_metrics.get('negative_recall'),
                        'positive_precision': avg_metrics.get('positive_precision'),
                        'negative_precision': avg_metrics.get('negative_precision'),
                        'auc_score': avg_metrics.get('auc_pr')
                    }
                    results.append(result)

    # Tworzenie DataFrame z wynikami
    df = pd.DataFrame(results)
    print(df)
    return df

process_files('./')