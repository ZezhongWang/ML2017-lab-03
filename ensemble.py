import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math


class AdaBoostClassifier(object):
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier=DecisionTreeClassifier, n_weakers_limit=5):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self, X, y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        # M 为分类器数目
        M = self.n_weakers_limit
        # N 为样本个数
        N = len(X)
        # w 为分类器权重
        w = np.zeros(M)
        # D 为样本权重
        D = np.array([1.0 / N for i in range(N)])
        clfs = []
        for i in range(M):
            clf = self.weak_classifier()
            clf.fit(X, y, sample_weight=D)
            # 计算错误率
            error_rate = max(1 - clf.score(X, y, sample_weight=D), 10 ** (-8))
            # 更新分类器权重 w
            w[i] = 0.5 * math.log((1 - error_rate) / error_rate)
            #   更新样本权重
            # calculate normalization factor
            y_pre = clf.predict(X)
            Z = 0
            for j in range(N):
                Z += D[j] * math.exp(-D[j] * y_pre[j] * y[j])
            # update sample weight
            for j in range(N):
                D[j] = D[j] * math.exp(-D[j] * y_pre[j] * y[j]) / Z
            clfs.append(clf)
        self.clfs = clfs
        self.w = w

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        y_predict_list = []
        for i in range(self.n_weakers_limit):
            y_predict_list.append(self.clfs.predict(X))
        y_predict_score = []
        for i in range(len(X)):
            s = 0
            for j in range(self.n_weakers_limit):
                s += w[j] * y_predict_list[i][j]
            y_predict.append(s)
        return y_predict_score

    def score(self, X, y):
        y_pre = self.predict(X)
        score = 0
        for i in range(len(X)):
            if y[i] == y_pre[i]:
                score += 1
        return float(score)/len(X)

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_predict_list = []
        for i in range(self.n_weakers_limit):
            y_predict_list.append(self.clfs[i].predict(X))
        y_predict = []
        for i in range(len(X)):
            s = 0
            for j in range(self.n_weakers_limit):
                s += self.w[j] * y_predict_list[j][i]
            if s > threshold:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        return y_predict

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
