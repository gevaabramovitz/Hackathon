import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from classifier import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


class DecisionTree:
    def __init__(self, maxDepth=None):
        """
        initializing decision tree and model to be trained.
        :param maxDepth: maximum depth of tree.
        """
        self.model = None
        self.tree = DecisionTreeClassifier(max_depth=maxDepth)

    def fit(self, x: np.array, y: np.array):
        """
        fit the model:
        :param x: dataset
        :param y: response vector
        :return: nothing, trains the self.model.
        """
        self.model = self.tree.fit(x, y)

    def predict(self, x: np.array):
        """
        gets test set and predicts the response vector.
        :param x: test set
        :return: predicted vector.
        """
        return self.tree.predict(x)


def score(model, x: np.array, y: np.array):
    """
    runs the model predition on x and compares to the real y
    :param model: one of the models
    :param x: test set
    :param y: real y
    :return: dictionary contains num of samples, fp, tp, acuuracy.
    """
    y_predict = model.predict(x)
    error, true_pos, true_neg, false_pos, false_neg, n, p = 0, 0, 0, 0, 0, 0, 0
    tp = [0, 0, 0, 0, 0]
    fp = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]
    for i in range(len(y)):
        if y_predict[i] != y[i]:
            error += 1
            fp[y[i]][y_predict[i]] += 1
        else:
            tp[y_predict[i]] += 1
    score_dict = {"num_samples": len(y), "fp table": fp, "tp table": tp,
                  "accuracy": sum(tp) / len(y)}
    return score_dict


def runTTmodel(model, X_train, Y_train, X_test, Y_test):
    """
    trains a model on X_train, Y_train and runs on X_test, Y_test.
    :param model: one of the our models.
    :param X_train: train set, numpy array.
    :param Y_train: trains response vector, numpy array.
    :param X_test: test set.
    :param Y_test: test response vector.
    :return: nothing.
    """
    model.fit(X_train, Y_train)
    score_dict = score(model, X_test, Y_test)
    print("false predictions: predicted column j while i is true label")
    fp = score_dict["fp table"]
    for i in range(len(fp)):
        print(fp[i])
    print("tp table", score_dict["tp table"])
    print("accuracy", score_dict["accuracy"])


def runTrees(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by random forest and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("Trees models")
    for i in range(1, 5):
        dt = DecisionTree(i, )
        print("Max depth:", i)
        runTTmodel(dt, X_train, Y_train, X_test, Y_test)


def runSVM(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by SVM and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("SVM models")
    types = ['linear', 'poly']
    multi_ways = ['ovo', 'ovr']
    for i in range(1, 5):
        for tpe in types:
            for mul_func in multi_ways:
                dt = SVC(i, ker=tpe, decision=mul_func)
                print("lmbda:", i, "svm type:", tpe, "multi:", mul_func)
                runTTmodel(dt, X_train, Y_train, X_test, Y_test)


def runKnn(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by K-neares-neighbors and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("KNN models")
    for i in (list(range(4, 50, 5)) + [2, 3]):
        print("number of neighbors:", i)
        kn = KNeighborsClassifier(n_neighbors=i)
        runTTmodel(kn, X_train, Y_train, X_test, Y_test)


def runLDA(X_train, Y_train, X_test, Y_test):
    """
    trains the train set by LDA and checks it by the test set.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    print("LDA model")
    set_lda = ['svd', 'lsqr']
    for ld in set_lda:
        print("LDA Model:", ld)
        lda = LinearDiscriminantAnalysis(solver=ld)
        runTTmodel(lda, X_train, Y_train, X_test, Y_test)


def runBag(X_train, Y_train, X_test, Y_test):
    # dt = DecisionTreeClassifier(max_depth=5)
    # runTTmodel(dt, X_train, Y_train, X_test, Y_test)
    print("Bag model")
    bm = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),
                           n_estimators=30)
    runTTmodel(bm, X_train, Y_train, X_test, Y_test)


def runTrainTest(X, Y, newX, newY):
    runTrees(X, Y, newX, newY)
    runKnn(X, Y, newX, newY)
    # runSVM(X, Y, newX, newY)
    runLDA(X, Y, newX, newY)


def baggingModels(models, X_train, Y_train, X_test, Y_test):
    accuracy_arr = np.zeros(len(models))
    print("Bagging")
    for i, model in enumerate(models):
        print("Bagging model:", model)
        bm = BaggingClassifier(base_estimator=model)
        accuracy_arr[i] = runTTmodel(bm, X_train, Y_train, X_test, Y_test)
    print(accuracy_arr)


def run_adaboost(X_train, Y_train, X_test, Y_test):
    """
    runs adaboost on the data.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=50)  # 0.477
    # model = AdaBoostClassifier(base_estimator=SVM(30))
    runTTmodel(model, X_train, Y_train, X_test, Y_test)


def learner_by_loc(X_train, Y_train, X_test, Y_test):
    """
    learning algorithm just by location.
    :param X_train: train set.
    :param Y_train: train response vector.
    :param X_test: test set.
    :param Y_test:test response vector.
    :return: nothing.
    """
    feats = ['Latitude', 'Longitude']
    for name in list(X_train.columns):
        if "half" in name:
            feats.append(name)
    X_train = X_train[feats]
    X_test = X_test[feats]
    X_train, Y_train = X_train.to_numpy(), Y_train.to_numpy()
    X_test, Y_test = X_test.to_numpy(), Y_test.to_numpy()
    model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=400), n_estimators=100)
    runTTmodel(model, X_train, Y_train, X_test, Y_test)


def plot_points_q_99(x_p, x_n, x_p1, x_n1, x5, ax, num_of_samples):
    ax.scatter(x_p.T[0], x_p.T[1], s=1, marker='.', c='blue')
    ax.scatter(x_n.T[0], x_n.T[1], s=1, marker='.', c='orange')
    ax.scatter(x_p1.T[0], x_p1.T[1], s=1, marker='.', c='red')
    ax.scatter(x_n1.T[0], x_n1.T[1], s=1, marker='.', c='green')
    ax.scatter(x5.T[0], x5.T[1], s=1, marker='.', c='pink')
    ax.set_title(f'Number of samples: {num_of_samples[0]}', size=15)
    ax.set_xlabel('Longitude', size=10)
    ax.set_ylabel('Latitude', size=10)


def showfigs(x, y, predicted_y):
    xlong = x['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = x['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_location = np.hstack((xlong, xlat))
    x_1 = x_location[y == 0]
    x_2 = x_location[y == 1]
    x_3 = x_location[y == 2]
    x_4 = x_location[y == 3]
    x_5 = x_location[y == 4]
    print("size of ", crimes_dict[0], "is: ", x_1.shape)
    print("size of ", crimes_dict[1], "is: ", x_2.shape)
    print("size of ", crimes_dict[2], "is: ", x_3.shape)
    print("size of ", crimes_dict[3], "is: ", x_4.shape)
    print("size of ", crimes_dict[4], "is: ", x_5.shape)
    print(x_location)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_points_q_99(x_1, x_2, x_3, x_4, x_5, ax, x_location.shape)
    ax.legend()
    fig.show()


def checkPicsLda(X, Y, X_test, Y_test):
    """
    show diffrences between the figures of the real ones
     and the predicted ones.
    """
    x = X
    X, Y = X.to_numpy(), Y.to_numpy()
    X_test, Y_test = X_test.to_numpy(), Y_test.to_numpy()
    ld = LinearDiscriminantAnalysis()
    ld.fit(X, Y)
    predY = ld.predict(X_test)
    showfigs(x, Y_test, predY)


def k_folds(k, X, y, models):
    """
    splits to k folds an the returns the best model and its average.
    :param k: number of folds.
    :param X: data set, numpy array.
    :param y: vector response, numpy array.
    :param models: differnt models for prdiction
    :return:
    """
    means = []
    for model in models:
        cross = cross_val_score(model, X, y, cv=k)
        print(cross)
        means.append(np.mean(cross))
    return np.argmax(np.array(means)), np.max(np.array(means))


def create_models():
    """
    creates different type of models to be tested in k-folds
    :return: the models, array.
    """
    models_chunks = []

    # trees
    temp = []
    for i in (list(range(2, 21, 7)) + [3, 4]):
        temp.append(BaggingClassifier(base_estimator=DecisionTreeClassifier(
            max_depth=i), n_estimators=6))
    models_chunks.append(temp)

    # k-nn
    temp = []
    for i in (list(range(4, 50, 22)) + [2, 5]):
        temp.append(BaggingClassifier(KNeighborsClassifier(n_neighbors=i),
                    n_estimators=2))
    models_chunks.append(temp)

    # lda
    temp = []
    set_lda = ['svd', 'lsqr']
    for ld in set_lda:
        temp.append(BaggingClassifier(LinearDiscriminantAnalysis(solver=ld),
                    n_estimators=10))
    models_chunks.append(temp)

    return models_chunks


def find_best(X, y):
    """
    runs the k-folds on different models.
    :param X: dataset
    :param y: response vector
    :return: nothing
    """
    models_chunks = create_models()
    for chunk in models_chunks:
        print(k_folds(5, X, y, chunk))

