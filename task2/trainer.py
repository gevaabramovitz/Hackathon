
from task2.classifier import *
from task2.models import *
from task2.police import *
PATH1 = '../Dataset_crimes.csv'
PATH2 = '../crimes_dataset_part2.csv'


def erase_wrong_lines(X):
    """
    erasing rows with wrong data.
    :param X: data set.
    :return: updated data set.
    """
    X.dropna(inplace=True)
    X.drop(X[((X["Community Area"] <= 0) | (X["Community Area"] > 77))].index,
           inplace=True)
    X.drop(X[((X['Latitude'] <= 41.5) | (X["Latitude"] > 42.1))].index,
           inplace=True)
    X.drop(X[((X["Longitude"] <= -88.0) | (X["Longitude"] > -87.5))].index,
           inplace=True)

    return X


def pre_process():
    """
    preproccessing the data.
    :return: data set, response vector, list of all the features.
    """
    first_data = pd.read_csv(PATH1, index_col=0)
    second_data = pd.read_csv(PATH2, index_col=0)
    print(first_data.shape)
    print(second_data.shape)
    print(first_data)

    X = first_data.append(second_data)

    print(X.shape)
    print(X)

    # line and column erasing
    X = erase_wrong_lines(X)

    X = remove_redundant_cols(X)

    # parse time - catch wrong formats
    X, times = get_parsed_time(X)

    X = create_features_related_to_time(X, times)

    # get dummies variables from: Beat, Location Description, Ward, Block:
    X = create_dummies(X)

    Y = X["Primary Type"]
    del X["Primary Type"]

    FEATURES = list(X.columns)

    return X, Y.apply(lambda x: crimes_dict_reverse[x]), FEATURES

def get_learner():
    """
    returns the decided model.
    :return: ...
    """
    return BaggingClassifier(LinearDiscriminantAnalysis(solver='lsqr'),
                                 n_estimators=10)
    # return LinearDiscriminantAnalysis(solver='lsqr')


if __name__ == '__main__':

    X, Y, FEATURES = pre_process()

    X_train, y_train= X[:32000], Y[:32000]
    X_test, y_test = X[32000:], Y[32000:]
    #train_police(X_train,X_test)
    print(FEATURES)

    learner = get_learner()

    learner.fit(X, Y)
#     X_test2 = pre_process_train('crimes_dataset_part2.csv')
#     real_y = pd.read_csv('crimes_dataset_part2.csv', index_col=0)['Primary Type']
#     real_y = real_y.apply(lambda x: crimes_dict_reverse[x])
#
#     # y = learner.predict(X_test2)
#     # print(y)
#     #print(np.sum(real_y != y) / 42780)
#
#
    with open(PKL_FILE, 'wb') as file:
        pickle.dump(learner, file)



