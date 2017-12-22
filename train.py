import pickle
import numpy as np
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import datetime

if __name__ == "__main__":
    # write your code here

    face_features = []
    nonface_features = []
    # 选择训练集, 测试集大小
    train_size = 400
    test_size = 50
    # load train data
    print('Loading Train data ......')
    with open('face.pickle', 'rb') as f:
        num = pickle.load(f)
        for _ in range(train_size+test_size):
            face_features.append(pickle.load(f))
    with open('nonface.pickle', 'rb') as f:
        num = pickle.load(f)
        for _ in range(train_size+test_size):
            nonface_features.append(pickle.load(f))
    print('Train data load finish')
    X_train = face_features[:train_size]
    X_train.extend(nonface_features[:train_size])
    y_train = [1 for _ in range(train_size)]
    y_train.extend([-1 for _ in range(train_size)])
    ada = AdaBoostClassifier(DecisionTreeClassifier)
    # train model
    print('Fit model')
    begin_time = datetime.datetime.now()
    ada.fit(X_train, y_train)
    end_time = datetime.datetime.now()
    print('Fit model finish, Train Time cost is: ', end_time - begin_time)
    del X_train, y_train
    # load test data
    print('Loading Test data ......')
    X_test = face_features[train_size:]
    X_test.extend(nonface_features[train_size:])
    y_test = [1 for _ in range(test_size)]
    y_test.extend([-1 for _ in range(test_size)])
    print('Test data load finish')
    # model predict
    print('Model predict ......')
    begin_time = datetime.datetime.now()
    y_pred = ada.predict(X_test)
    end_time = datetime.datetime.now()
    print('Predict Time cost is: ', end_time - begin_time)
    print('************************************')
    print(classification_report(y_test, y_pred, target_names=['face', 'non face']))
