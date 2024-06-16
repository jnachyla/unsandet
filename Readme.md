## Instrukcja do Projektu

### Instalacja

1. Zainstaluj wymagane biblioteki za pomocą pliku `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
### Zawartość 

### Zawartość Plików w Projekcie

- `results/`: Katalog, w którym zapisywane są wyniki eksperymentów w formacie JSON.
- `ad_meta_cost.py`: Implementacja algorytmu MetaCost.
- `ad_one_class.py`: Implementacja algorytmów klasyfikacji jednoklasowej.
- `anomaly_detector.py`: Implementacja klasy AnomalyDetector do detekcji anomalii za pomocą algorytmów grupowania.
- `data_prep.py`: Funkcje pomocnicze do przygotowania i podziału danych.
- `experiments.py`: Skrypt zawierający definicje i uruchamianie eksperymentów.
- `http_eval.csv`: Dane ewaluacyjne dla zbioru HTTP.
- `http_train.csv`: Dane treningowe dla zbioru HTTP.
- `metrics.py`: Funkcje obliczające metryki ewaluacyjne dla modeli.
- `Readme.md`: Plik z instrukcjami i opisem projektu.
- `reports.py`: Skrypt do generowania raportu z plików JSON.
- `requirements.txt`: Lista zależności potrzebnych do uruchomienia projektu.
- `shuttle.mat`: Dane dla zbioru Shuttle w formacie MAT.
- `shuttle_eval.csv`: Dane ewaluacyjne dla zbioru Shuttle.
- `shuttle_train.csv`: Dane treningowe dla zbioru Shuttle.
- `subsampler.py`: Funkcje do przeprowadzania subsamplingu danych.
- `test_ad_one_class.py`: Skrypt testujący algorytmy klasyfikacji jednoklasowej.
- `zum_notebook.ipynb`: Notebook zawierający dokumentację i szczegóły eksperymentów.

### Uruchamianie Eksperymentów

Eksperymenty znajdują się w pliku `experiments.py`. Aby je uruchomić i zapisać wyniki do plików JSON w katalogu `results`, wykonaj poniższe kroki:

1. Otwórz plik `experiments.py` i odkomentuj linie dotyczące eksperymentów, które chcesz uruchomić. Na przykład:
    ```
   # exps.run_http_one_class()
   #exps.run_shuttle_one_class()
   #
   # exps.run_http_meta_cost()
   # exps.run_shuttle_meta_cost()
   #
   # exps.run_http_dbscan_experiment()
   #exps.run_http_kmeans_experiment()
   # exps.run_http_agglomerative_experiment()
   #
   # exps.run_shuttle_dbscan_experiment()
   # exps.run_shuttle_kmeans_experiment()
   # exps.run_shuttle_agglomerative_experiment()
    ```
2. Uruchom skrypt `experiments.py`:
    ```bash
    python experiments.py
    ```
3. Wyniki eksperymentów zostaną zapisane jako pliki JSON w katalogu `results`.

### Dokumentacja

Dokumentacja projektu znajduje się w notebooku `zum_notebook.ipynb`. Notebook ten zawiera szczegółowe opisy eksperymentów, metodyki oraz wyniki.

Aby otworzyć notebooka, możesz skorzystać z Google Colab lub innego środowiska Jupyter Notebook:

1. Przejdź do katalogu projektu:
    ```bash
    cd repo
    ```
2. Uruchom notebooka:
    ```bash
    jupyter notebook zum_notebook.ipynb
    ```
    Lub zaimportuj notebooka do Google Colab i otwórz go tam.