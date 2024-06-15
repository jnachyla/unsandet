from typing import Optional
import numpy as np
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

class AnomalyDetector:
    def __init__(
            self,
            model: str = "kmeans",
            metric: str = "euclidean",
            n_clusters: Optional[int] = None,
    ) -> None:
        if model not in ["kmeans", "dbscan", "agglomerative"]:
            raise ValueError("Unknown model.")
        if metric not in ["euclidean", "cityblock", "mahalanobis"]:
            raise ValueError("Unknown metric.")
        if n_clusters is not None and model == "dbscan":
            raise ValueError("DBSCAN does not use n_clusters as a parameter!")

        self.model_name = model
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

    def fit_predict(self, data):
        data_scaled = self.scaler.fit_transform(data)

        if self.model_name == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters)
            labels = self.model.fit_predict(data_scaled)
            centers = self.model.cluster_centers_
        if self.model_name == "dbscan":
            self.model = DBSCAN(metric=self.metric)
            labels = self.model.fit_predict(data_scaled)
            centers = None
        if self.model_name == "agglomerative":
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=None, connectivity=None)
            labels = self.model.fit_predict(data_scaled)
            centers = np.array([data_scaled[labels == i].mean(axis=0) for i in range(max(labels) + 1)])

        if centers is not None:
            distances = self._compute_distances(data_scaled, centers)
        else:
            distances = self._handle_dbscan_distances(data_scaled, labels)

        return labels, distances

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
            distances = np.min(pairwise_distances(data, centers, metric="euclidean"), axis=1)

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


class MetaCost:
    def __init__(self, base_detector: AnomalyDetector, cost_matrix: np.ndarray, m: int, n: int, p: bool = False, q: bool = True):
        """
        Inicjalizuje instancję klasy MetaCost.

        Parametry:
        base_detector: AnomalyDetector
            Obiekt klasy AnomalyDetector, który będzie używany jako podstawowy model detekcji anomalii.
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
        """
        self.base_detector = base_detector
        self.cost_matrix = cost_matrix
        self.m = m
        self.n = n
        self.p = p
        self.q = q

    def fit_predict(self, data):
        self.base_detector.fit(data)
        self.models = []
        self.samples = []

        for _ in range(self.m):
            sample = resample(data, n_samples=self.n)
            model = AnomalyDetector(model=self.base_detector.model_name, metric=self.base_detector.metric, n_clusters=self.base_detector.n_clusters)
            model.fit(sample)
            self.models.append(model)
            self.samples.append(sample)

        probabilities = self._calculate_probabilities(data)
        assigned_clusters = self._assign_clusters(data, probabilities)

        final_model = AnomalyDetector(model=self.base_detector.model_name, metric=self.base_detector.metric, n_clusters=self.base_detector.n_clusters)
        final_model.fit(data)
        final_model.model.fit(data, assigned_clusters)

        return assigned_clusters

    def _calculate_probabilities(self, data):
        n_samples, n_clusters = data.shape[0], self.base_detector.n_clusters
        probabilities = np.zeros((n_samples, n_clusters))

        for j, x in enumerate(data):
            relevant_models = range(len(self.models)) if self.q else [i for i in range(len(self.models)) if x not in self.samples[i]]
            for i in relevant_models:
                model = self.models[i]
                if self.p:
                    probabilities[j] += model.model.predict_proba([x])[0]
                else:
                    labels = model.model.predict([x])
                    probabilities[j, labels[0]] += 1

        for i in range(n_samples):
            probabilities[i] /= np.sum(probabilities[i])

        return probabilities

    def _assign_clusters(self, data, probabilities):
        n_samples, n_clusters = probabilities.shape
        assigned_clusters = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            cost = [np.sum(probabilities[i] * self.cost_matrix[:, j]) for j in range(n_clusters)]
            assigned_clusters[i] = np.argmin(cost)

        return assigned_clusters


# Przykład użycia
def test():
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

    detector = AnomalyDetector(model="kmeans", metric="euclidean", n_clusters=2)
    cost_matrix = np.array([[0, 1], [1, 0]])

    meta_cost = MetaCost(base_detector=detector, cost_matrix=cost_matrix, m=10, n=80)
    meta_cost.fit(X)

    labels, distances = detector.fit_predict(X)

    print("Ostateczne środki klastrów:")
    print(detector.centers)
    print("Przypisane klastry dla każdego przykładu:")
    print(labels)
