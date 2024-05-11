from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

class OneClassAnnomalyDetector:
    def __init__(self, model_name = "oneclasssvm"):
        if model_name not in ["oneclasssvm", "isolationforest"]:
            raise ValueError("Unknown model.")

        if model_name == "oneclasssvm":
            self.model = OneClassSVM()
        else:
            self.model = IsolationForest()

    def fit(self, X_train):
        self.model.fit(X_train)


    def predict(self, Xtest):
        ypred = self.model.predict(Xtest)

        return ypred