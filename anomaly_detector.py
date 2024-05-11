import numpy as np
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    def __init__(
        self,
        model: str = "kmeans",
        metric: str = "euclidean",
        n_clusters: int | None = None,
    ) -> None:
        if model not in ["kmeans", "dbscan", "agglomerative"]:
            raise ValueError("Unknown model.")
        if metric not in ["euclidean", "cityblock", "mahalanobis"]:
            raise ValueError("Unknown metric.")
        if n_clusters is not None and model == "dbscan":
            raise ValueError("DBSCAN does not use n_clusters as a parameter!")

        self.model = model
        self.metric = metric
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters

    def fit_predict(self, data):
        data_scaled = self.scaler.fit_transform(data)

        if self.model == "kmeans":
            model = KMeans(n_clusters=self.n_clusters)
            labels = model.fit_predict(data_scaled)
            centers = model.cluster_centers_
        if self.model == "dbscan":
            model = DBSCAN(metric=self.metric)
            labels = model.fit_predict(data_scaled)
            centers = None
        if self.model == "agglomerative":
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters, distance_threshold=None, connectivity=None
            )
            labels = model.fit_predict(data_scaled)
            centers = np.array(
                [data_scaled[labels == i].mean(axis=0) for i in range(max(labels) + 1)]
            )

        if centers is not None:
            distances = self.compute_distances(data_scaled, centers)
        else:
            distances = self.handle_dbscan_distances(data_scaled, labels)

        return labels, distances

    def handle_dbscan_distances(self, data: np.ndarray, labels):
        noise_indexes = labels == -1
        distances = np.zeros(data.shape[0])
        distances[noise_indexes] == np.inf

        for label in np.unique(labels[labels != -1]):
            cluster_indexes = labels == label
            cluster_points = data[cluster_indexes]
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(cluster_points)
            distances_cluster = nn.kneighbors(
                cluster_points, n_neighbors=2, return_distance=True
            )[0][:, 1]
            distances[cluster_indexes] = distances_cluster

        return distances

    def compute_distances(self, data, centers):
        if self.metric == "mahalanobis":
            covariance_matrix = EmpiricalCovariance().fit(data).covariance_
            inv_covariance_matrix = np.linalg.inv(covariance_matrix)
            distances = [
                mahalanobis(x, centers[cluster], inv_covariance_matrix)
                for x, cluster in zip(
                    data, np.argmin(cdist(data, centers, "mahalanobis"), axis=1)
                )
            ]
        if self.metric == "cityblock":
            distances = np.min(cdist(data, centers, metric="cityblock"), axis=1)
        if self.metric == "euclidean":
            distances = np.min(
                pairwise_distances(data, centers, metric="euclidean"), axis=1
            )

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
