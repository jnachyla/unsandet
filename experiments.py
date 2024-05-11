import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score

import data_prep
from ad_one_class import OneClassAnnomalyDetector


class Experiments:
    def __init__(self):
        self.http_dataset = self.load_data_http()
        self.shuttle_dataset = self.load_data_shuttle()
        pass
    def load_data_http(self):


        X = pd.read_csv("http_train.csv",header=0).values
        y = pd.read_csv("http_eval.csv",header=0).values

        return (X,y)



    def load_data_shuttle(self):
        X = pd.read_csv("shuttle_train.csv",header=0).values
        y = pd.read_csv("shuttle_eval.csv",header=0).values

        return (X,y)


    def run_http(self):
        Xtrain, ytrain, Xtest, ytest = data_prep.split_binary_dataset(self.http_dataset[0], self.http_dataset[1])

        #fit isolation forest
        print("Fitting Isolation Forest...")
        #print shape of Xtrain with name formated
        print(f"Shape of Xtrain: {Xtrain.shape}")
        isolation_forest = OneClassAnnomalyDetector(model_name = "isolationforest")
        isolation_forest.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_forest = isolation_forest.predict(Xtest)
        print("Results: HTTP one class IsolationForest")
        self.print_metrics(ytest, ypred_forest)

        svm = OneClassAnnomalyDetector(model_name = "oneclasssvm")
        print("Fitting Model...")

        svm.fit(Xtrain)
        print("Fitted. Predicting...")
        ypred_svm = svm.predict(Xtest)

        print("Results: HTTP one class SVM")
        self.print_metrics(ytest, ypred_svm)

    def print_metrics(self, Ytest, ypred):
        print(ypred)
        # show fraction of outliers
        print("Outliers predicted fraction:")
        print(sum(ypred) / len(ypred))
        print("Metrics")
        # compute accuracy and recall precision using sklearn metrics with printed names
        print("Accuracy")
        print(accuracy_score(Ytest, ypred))
        print("Recall")
        print(recall_score(Ytest, ypred))
        print("Precision")
        print(precision_score(Ytest, ypred))






