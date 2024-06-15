import time
from typing import Optional

import pandas as pd
import tqdm
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min

class AnomalyDetector:
    def __init__(
            self,
            model_name: str = "kmeans",
            metric: str = "euclidean",
            n_clusters: Optional[int] = None,
    ) -> None:
        if model_name not in ["kmeans", "dbscan", "agglomerative"]:
            raise ValueError("Unknown model.")
        if metric not in ["euclidean", "cityblock", "mahalanobis"]:
            raise ValueError("Unknown metric.")
        if n_clusters is not None and model_name == "dbscan":
            raise ValueError("DBSCAN does not use n_clusters as a parameter!")

        self.model_name = model_name
        self.metric = metric
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters

    def fit(self, data):
        data_scaled = self.scaler.fit_transform(data)

        if self.model_name == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters)
            self.model.fit(data_scaled)
            self.centers = self.model.cluster_centers_
        elif self.model_name == "dbscan":
            self.model = DBSCAN(metric=self.metric)
            self.model.fit(data_scaled)
            self.centers = None
        elif self.model_name == "agglomerative":
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=None, connectivity=None)
            self.model.fit(data_scaled)
            self.centers = np.array(
                [data_scaled[self.model.labels_ == i].mean(axis=0) for i in range(max(self.model.labels_) + 1)]
            )

    def predict(self, data):
        """
        Przewiduje punkty najbliższe do centrów klastrów.

        Parametry:
        data: np.ndarray
            Zbiór danych do przewidywania.

        Zwraca:
        np.ndarray: Indeksy najbliższych punktów do centrów klastrów.
        """
        data_scaled = self.scaler.transform(data)

        if self.centers is not None:
            closest_points = self._compute_distances_all(data_scaled, self.centers)
        else:

            closest_points = self._compute_distances_all(data_scaled, self.model.labels_)

        classes = np.argmin(closest_points, axis=1)
        #if closes points is scalar, convert to array
        if not isinstance(classes, np.ndarray):
            classes = np.array([classes])

        labels = AnomalyDetector.transform_labels(classes)

        return labels


    def fit_predict(self, data):
        transform = self.scale_features(data)
        data_scaled = transform

        if self.model_name == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters)
            labels = self.model.fit_predict(data_scaled)
            self.centers = self.model.cluster_centers_
        if self.model_name == "dbscan":
            self.model = DBSCAN(metric=self.metric)
            labels = self.model.fit_predict(data_scaled)
            self.centers = None
        if self.model_name == "agglomerative":
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=None, connectivity=None)
            labels = self.model.fit_predict(data_scaled)
            self.centers = np.array([data_scaled[labels == i].mean(axis=0) for i in range(max(labels) + 1)])

        if self.centers is not None:
            distances = self._compute_distances(data_scaled, self.centers)
        else:
            distances = self._handle_dbscan_distances(data_scaled, labels)

        return labels, distances

    def scale_features(self, data):
        return self.scaler.fit_transform(data)

    def _handle_dbscan_distances(self, data: np.ndarray, labels):
        noise_indexes = labels == -1
        distances = np.zeros(data.shape[0])
        distances[noise_indexes] = np.inf

        for label in np.unique(labels[labels != -1]):
            cluster_indexes = labels == label
            cluster_points = data[cluster_indexes]
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(cluster_points)
            distances_cluster = nn.kneighbors(cluster_points, n_neighbors=2, return_distance=True)[0][:, 1]
            distances[cluster_indexes] = distances_cluster

        return distances

    def _compute_distances(self, data, centers):
        if self.metric == "mahalanobis":
            covariance_matrix = EmpiricalCovariance().fit(data).covariance_
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            distances = [
                mahalanobis(x, centers[cluster], inv_covariance_matrix)
                for x, cluster in zip(data, np.argmin(cdist(data, centers, "mahalanobis"), axis=1))
            ]
        if self.metric == "cityblock":
            distances = np.min(cdist(data, centers, metric="cityblock"), axis=1)
        if self.metric == "euclidean":
            distances = np.min(pairwise_distances(data, centers, metric="euclidean", n_jobs=-1), axis=1)

        return distances

    def _compute_distances_all(self, data, centers):
        if self.metric == "mahalanobis":
            covariance_matrix = EmpiricalCovariance().fit(data).covariance_
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            distances = [
                mahalanobis(x, centers[cluster], inv_covariance_matrix)
                for x, cluster in zip(data, cdist(data, centers, "mahalanobis"))
            ]
        if self.metric == "cityblock":
            distances = cdist(data, centers, metric="cityblock")
        if self.metric == "euclidean":
            distances = pairwise_distances(data, centers, metric="euclidean", n_jobs=-1)

        return distances

    @staticmethod
    def detect_anomalies(distances, threshold: float = 0.95):
        threshold_value = np.quantile(distances, threshold)
        anomalies = np.where(distances > threshold_value)[0]
        return anomalies

    @staticmethod
    def transform_distances(distances, threshold=0.95):
        transformed_distances = np.zeros_like(distances)
        anomalies = AnomalyDetector.detect_anomalies(distances, threshold)
        transformed_distances[anomalies] = 1
        return transformed_distances

    @staticmethod
    def transform_labels(labels):
        unique_labels, counts = np.unique(labels, return_counts=True)
        most_populous_label = unique_labels[np.argmax(counts)]
        transformed_labels = np.where(labels != most_populous_label, 1, 0)

        return transformed_labels


def validate_rf(X, y):
    # test classication on rf
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    print(f"RF - Precision: {precision:.2f}")
    print(f"RF - Recall: {recall:.2f}")
    print(f"RF - F1 Score: {f1:.2f}")


class MetaCost:
    def __init__(self, base_detector, cost_matrix, m, n, p=False, q=True):
        """
        Inicjalizuje instancję klasy MetaCost.

        Parametry:
        base_detector: KMeans
            Obiekt klasy KMeans, który będzie używany jako podstawowy model detekcji anomalii.
        cost_matrix: np.ndarray
            Macierz kosztów, która określa koszty błędnej klasyfikacji między różnymi klasami.
        m: int
            Liczba resamplowanych próbek, które mają być wygenerowane.
        n: int
            Liczba przykładów w każdej resamplowanej próbce.
        p: bool (domyślnie False)
            Jeżeli `True`, algorytm klasyfikacyjny zwraca prawdopodobieństwa klas.
            Jeżeli `False`, zakłada, że przypisanie do klasy jest binarne (1 dla przypisanej klasy, 0 dla innych).
        q: bool (domyślnie True)
            Jeżeli `True`, wszystkie resample (próbki) są używane do obliczania prawdopodobieństw dla każdego przykładu.
            Jeżeli `False`, uwzględniane są tylko te modele, które zawierają punkt x w swoich próbkach.

        Pseudokod:
        Inputs:
        S is the training set,
        L is a classification learning algorithm,
        C is a cost matrix,
        m is the number of resamples to generate,
        n is the number of examples in each resample,
        p is True iff L produces class probabilities,
        q is True iff all resamples are to be used for each example.

        Procedure MetaCost (S, L, C, m, n, p, q)
        For i = 1 to m
            Let Si be a resample of S with n examples.
            Let Mi = Model produced by applying L to Si.
        For each example x in S
            For each class j
                Let P(j|x) = 1/SUMi ∑i P(j|x, Mi)
                Where
                    If p then P(j|x, Mi) is produced by Mi
                    Else P(j|x, Mi) = 1 for the class predicted by Mi for x, and 0 for all others.
                    If q then i ranges over all Mi
                    Else i ranges over all Mi such that x ∉ Si.
            Let x's class = argmini ∑j P(j|x)C(i, j).
        Let M = Model produced by applying L to S.
        Return M.
        """
        self.base_detector = base_detector
        self.cost_matrix = cost_matrix
        self.m = m
        self.n = n
        self.p = p
        self.q = q

    def fit_predict(self, data, labels):
        """
        Trenuje model MetaCost na danych i zwraca przypisania do klas.

        Parametry:
        data: np.ndarray
            Zbiór danych do trenowania modelu.

        Zwraca:
        np.ndarray: Przypisania do klas dla każdego przykładu w zbiorze danych.

        Pseudokod:
        For i = 1 to m
            Let Si be a resample of S with n examples.
            Let Mi = Model produced by applying L to Si.
        """
        self.models = []
        self.samples = []

        for _ in tqdm.tqdm(range(self.m)):
            X_sampled, y_sampled = self.startified_resample(data, labels, self.n)

            model = AnomalyDetector(n_clusters=self.base_detector.n_clusters, metric=self.base_detector.metric, model_name="kmeans")

            model.fit_predict(X_sampled)

            self.models.append(model)
            self.samples.append(X_sampled)

        #add time seconds measuremnt here
        print("Calculating probabilities and assigning clusters...")

        probabilities = self._calculate_probabilities(data)




        print("Assigning clusters...")
        assigned_clusters = self._assign_clusters(data, probabilities)



        final_model = KMeans(n_clusters=self.base_detector.n_clusters)
        final_model.fit(data)
        final_model.fit(data, assigned_clusters)

        return assigned_clusters

    def startified_resample(self, X, y, n):

        resampled_size = n/X.shape[0]
        X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=resampled_size)

        return X_test, y_test

    def _calculate_probabilities(self, data):
        """
        Oblicza prawdopodobieństwa przynależności do klas dla każdego przykładu.

        Parametry:
        data: np.ndarray
            Zbiór danych do trenowania modelu.

        Zwraca:
        np.ndarray: Prawdopodobieństwa przynależności do klas dla każdego przykładu.
        """
        n_samples, n_clusters = data.shape[0], self.base_detector.n_clusters
        probabilities = np.zeros((n_samples, n_clusters))

        if self.q:
            relevant_models = range(len(self.models))
        else:
            relevant_models = [i for i in range(len(self.models)) if any(x not in self.samples[i] for x in data)]

        for i in relevant_models:
            model = self.models[i]
            labels = model.predict(data)
            for j in range(n_samples):
                probabilities[j, labels[j]] += 1

        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def _assign_clusters(self, data, probabilities):
        """
        Przypisuje klastry na podstawie minimalnego kosztu.

        Parametry:
        data: np.ndarray
            Zbiór danych do trenowania modelu.
        probabilities: np.ndarray
            Prawdopodobieństwa przynależności do klas dla każdego przykładu.

        Zwraca:
        np.ndarray: Przypisania do klas dla każdego przykładu w zbiorze danych.

        Pseudokod:
        Let x's class = argmini ∑j P(j|x)C(i, j).
        """
        n_samples, n_clusters = probabilities.shape
        assigned_clusters = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            cost = [np.sum(probabilities[i] * self.cost_matrix[:, j]) for j in range(n_clusters)]
            assigned_clusters[i] = np.argmin(cost)

        return assigned_clusters


def generate_fixed_interval_cost_matrix(n_classes):
    """
    Generuje macierz kosztów dla modelu Fixed-Interval Cost.

    Parametry:
    n_classes: int
        Liczba klas.

    Zwraca:
    np.ndarray: Macierz kosztów o wymiarach (n_classes, n_classes).
    """
    C = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        C[i, i] = np.random.uniform(0, 1000)
        for j in range(n_classes):
            if i != j:
                C[i, j] = np.random.uniform(0, 10000)
    return C


def generate_probability_dependent_cost_matrix(n_classes, class_probabilities):
    """
    Generuje macierz kosztów dla modelu Probability-Dependent Cost.

    Parametry:
    n_classes: int
        Liczba klas.
    class_probabilities: np.ndarray
        Prawdopodobieństwa wystąpienia każdej klasy w zbiorze treningowym.

    Zwraca:
    np.ndarray: Macierz kosztów o wymiarach (n_classes, n_classes).
    """
    C = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        C[i, i] = np.random.uniform(0, 1000)
        for j in range(n_classes):
            if i != j:
                C[i, j] = np.random.uniform(0, 2000 * class_probabilities[i] / class_probabilities[j])
    return C


def evaluate_model(X,y, cost_matrix_generator = "fixed_interval", N=2, cost_matrix=None):
    """
    Ocena modelu MetaCost na zbiorze treningowym i testowym.

    Parametry:
    X_train: np.ndarray
        Zbiór danych treningowych.
    y_train: np.ndarray
        Etykiety danych treningowych.
    cost_matrix_generator: function
        Funkcja generująca macierz kosztów.

    Zwraca:
    tuple: Średnie wartości metryk Precision, Recall, F1 Score oraz Accuracy.
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    for _ in tqdm.tqdm(range(N)):
        class_probabilities = np.bincount(y) / len(y)
        if cost_matrix is not None:
            cost_matrix = cost_matrix
        elif cost_matrix_generator == "fixed_interval":
            cost_matrix = generate_fixed_interval_cost_matrix(2)
        else:
            cost_matrix = generate_probability_dependent_cost_matrix(2, class_probabilities)

        detector = AnomalyDetector(n_clusters=2, metric="euclidean", model_name="kmeans")

        n = X.shape[0] // 10

        meta_cost = MetaCost(base_detector=detector, cost_matrix=cost_matrix, m=30, n=1000)

        # Trenowanie modelu i przypisanie klastrów
        assigned_clusters = meta_cost.fit_predict(X,y )

        # Mapowanie klastrów na oryginalne klasy (zakładamy, że klasa 0 jest dominująca)
        # Można również użyć bardziej zaawansowanej logiki do mapowania klastrów na klasy
        y_pred = AnomalyDetector.transform_labels(assigned_clusters)


        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        accuracy = accuracy_score(y, y_pred)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    return avg_precision, avg_recall, avg_f1, avg_accuracy


def test_shuttle():


    X = pd.read_csv("shuttle_train.csv", header=0).values
    y = pd.read_csv("shuttle_eval.csv", header=0).values.ravel()
    validate_rf(X, y)

    cost_matrix = np.array([
        [0, 1],  # Koszty dla punktów normalnych (klasa 0)
        [5, 0]  # Koszty dla outlierów (klasa 1)
    ])
    # Ocena modelu z Fixed-Interval Cost Matrix
    avg_precision_fixed, avg_recall_fixed, avg_f1_fixed, avg_accuracy_fixed = evaluate_model(X,y,
                                                                                             "fixed_interval", N=5, cost_matrix=cost_matrix)
    print(f"Fixed-Interval Cost Matrix - Average Precision: {avg_precision_fixed:.2f}")
    print(f"Fixed-Interval Cost Matrix - Average Recall: {avg_recall_fixed:.2f}")
    print(f"Fixed-Interval Cost Matrix - Average F1 Score: {avg_f1_fixed:.2f}")
    print(f"Fixed-Interval Cost Matrix - Average Accuracy: {avg_accuracy_fixed:.2f}")



def test():
    # Generowanie syntetycznego zbioru danych
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               random_state=42, n_classes=2)

    # Podział zbioru danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Definicja klasy AnomalyDetector (implementacja jest taka sama jak wcześniej, zakładamy, że już jest zaimportowana)

    # Definicja klasy MetaCost (implementacja jest taka sama jak wcześniej, zakładamy, że już jest zaimportowana)

    # Konfiguracja detektora anomalii
    detector = AnomalyDetector(model="kmeans", metric="euclidean", n_clusters=2)
    cost_matrix = np.array([[0, 1], [1, 0]])

    # Inicjalizacja modelu MetaCost
    meta_cost = MetaCost(base_detector=detector, cost_matrix=cost_matrix, m=10, n=80)

    # Trenowanie modelu i przypisanie klastrów
    assigned_clusters = meta_cost.fit_predict(X_train)

    # Mapowanie klastrów na oryginalne klasy (zakładamy, że klasa 0 jest dominująca)
    # Można również użyć bardziej zaawansowanej logiki do mapowania klastrów na klasy
    y_pred = AnomalyDetector.transform_labels(assigned_clusters)

    # Obliczenie metryk
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)



    # Wyświetlenie wyników
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
