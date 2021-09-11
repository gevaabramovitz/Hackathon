
from classifier import *
import numpy as np


def get_distance_by_long_lat(long1, lat1, long2, lat2):
    R = 6373.0  # radius of the Earth

    lat1 = math.radians(lat1)
    lon1 = math.radians(long1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(long2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def get_is_close(xlat, xlong):
    R = 6373.0

    xlat_mul = np.repeat(xlat, [xlat.shape[0]], axis=1)
    xlat_mul2 = np.repeat(xlat.T, [xlat.shape[0]], axis=0)
    xlong_mul = np.repeat(xlong, [xlong.shape[0]], axis=1)
    xlong_mul2 = np.repeat(xlong.T, [xlong.shape[0]], axis=0)

    lat1 = np.radians(xlat_mul)
    lon1 = np.radians(xlong_mul)
    lat2 = np.radians(xlat_mul2)
    lon2 = np.radians(xlong_mul2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.power(np.sin(dlat / 2), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2), 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return (distance <= 0.5)


def correct(point, point_to_check):
    indic_t = abs(point[2] - point_to_check[2])
    if indic_t <= 30 or indic_t >= 1409:
        if get_distance_by_long_lat(point[0], point[1], point_to_check[0], point_to_check[1]) <= 0.5:
            return True
    return False

    # p1_h = point[3]
    # p2_h = point_to_check[3]
    # p1_m = point[2]
    # p2_m = point_to_check[2]
    # p1_lo = point[0]
    # p1_la = point[1]
    # p2_lo = point_to_check[0]
    # p2_la = point_to_check[1]
    # if p1_h - p2_h == 1 or p1_h - p2_h == -23:  # means we over the point_t_c time
    #     if p2_m - p1_m >= 30:
    #         if get_distance_by_long_lat(p1_lo, p1_la, p2_lo, p2_la) <= 0.5:
    #             return True
    #
    # elif p1_h - p2_h == -1 or p1_h - p2_h == 23:
    #     if p1_m - p2_m >= 30:
    #         if get_distance_by_long_lat(p1_lo, p1_la, p2_lo, p2_la) <= 0.5:
    #             return True
    #
    # elif p1_h - p2_h == 0:
    #     if abs(p2_m - p1_m) <= 30:
    #         if get_distance_by_long_lat(p1_lo, p1_la, p2_lo, p2_la) <= 0.5:
    #             return True
    #
    # return False


def pre_process_police(X):
    xlong = X['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))
    return np.hstack((xlong, xlat, x_time))


def pre_process_police_weekend(X):
    X = X[(X["weekday 4"] == 1) | (X["weekday 5"] == 1) | (X["weekday 6"] == 1)]
    xlong = X['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))
    return np.hstack((xlong, xlat, x_time))


def pre_process_police_not_weekend(X):
    X = X[(X["weekday 0"] == 1) | (X["weekday 1"] == 1) | (X["weekday 2"] == 1) | (X["weekday 3"] == 1)]
    xlong = X['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))
    return np.hstack((xlong, xlat, x_time))


def make_matrix_to_police(X):
    xlong = X['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))

    # xlong = X[:, 0]
    # xlat = X[:, 1]
    # x_time = X[:, 2]

    X_time_mul = np.repeat(x_time, [x_time.shape[0]], axis=1)
    X_time_mul2 = np.repeat(x_time.T, [x_time.shape[0]], axis=0)
    time_dif = X_time_mul - X_time_mul2
    time_dif = np.abs(time_dif)

    time_dif = (time_dif <= 30) | (time_dif >= 1409)

    is_dist = get_is_close(xlat, xlong)

    all_req = time_dif & is_dist
    res = np.count_nonzero(all_req, axis=0) - 1
    res = np.reshape(res, (-1, 1))
    index = np.arange(len(X))
    index = np.reshape(index, (-1, 1))

    return np.hstack((xlong, xlat, x_time, index, res))


def make_matrix_to_police_not_weekend(X):
    # add index col
    # index = list(range(len(X)))
    # index = np.reshape(index, (-1, 1))
    # X["index"] = index
    X_not_weekend = X[(X["weekday 0"] == 1) | (X["weekday 1"] == 1) | (X["weekday 2"] == 1) | (X["weekday 3"] == 1)]
    # filter weekend
    xlong = X_not_weekend['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X_not_weekend['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X_not_weekend['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))
    xin = X_not_weekend['index'].to_numpy()
    xin = np.reshape(xin, (-1, 1))
    # xlong = X[:, 0]
    # xlat = X[:, 1]
    # x_time = X[:, 2]

    X_time_mul = np.repeat(x_time, [x_time.shape[0]], axis=1)
    X_time_mul2 = np.repeat(x_time.T, [x_time.shape[0]], axis=0)
    time_dif = X_time_mul - X_time_mul2
    time_dif = np.abs(time_dif)

    time_dif = (time_dif <= 30) | (time_dif >= 1409)

    is_dist = get_is_close(xlat, xlong)

    all_req = time_dif & is_dist
    res = np.count_nonzero(all_req, axis=0) - 1
    res = np.reshape(res, (-1, 1))
    # index = np.arange(len(X))
    # index = np.reshape(index, (-1, 1))
    return np.hstack((xlong, xlat, x_time, xin, res))


def make_matrix_to_police_weekend(X):
    # add index col
    # index = list(range(len(X)))
    # # index = np.reshape(index, (-1, 1))
    # X["index"] = index
    X_weekend = X[(X["weekday 4"] == 1) | (X["weekday 5"] == 1) | (X["weekday 6"] == 1)]
    # filter weekend
    xlong = X_weekend['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X_weekend['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time = X_weekend['time'].to_numpy().astype(int)
    x_time = np.reshape(x_time, (-1, 1))
    xin = X_weekend['index'].to_numpy()
    xin = np.reshape(xin, (-1, 1))

    # xlong = X[:, 0]
    # xlat = X[:, 1]
    # x_time = X[:, 2]

    X_time_mul = np.repeat(x_time, [x_time.shape[0]], axis=1)
    X_time_mul2 = np.repeat(x_time.T, [x_time.shape[0]], axis=0)
    time_dif = X_time_mul - X_time_mul2
    time_dif = np.abs(time_dif)

    time_dif = (time_dif <= 30) | (time_dif >= 1409)

    is_dist = get_is_close(xlat, xlong)

    all_req = time_dif & is_dist
    res = np.count_nonzero(all_req, axis=0) - 1
    res = np.reshape(res, (-1, 1))
    # index = np.arange(len(X))
    # index = np.reshape(index, (-1, 1))
    return np.hstack((xlong, xlat, x_time, xin, res))


def collides(point, arr_points):
    for ps in arr_points:
        if correct(point, ps):
            return True
    return False


def choose30(arr_point):
    chosen = []
    counter = 0
    i = len(arr_point) - 1
    while counter < 30:
        if not collides(arr_point[i], chosen):
            chosen.append(arr_point[i])
            counter += 1
        i -= 1
    return chosen


def testResult(chosen, X_test):
    counter = 0
    for i in range(len(X_test)):
        if collides(X_test[i], chosen):
            counter += 1
    return counter / len(X_test)


def testResultWeekend(chosen, X_test):
    counter = 0
    for i in range(len(X_test)):
        if collides(X_test[i], chosen):
            counter += 1
    return counter / len(X_test)


def testResultNotWeekend(chosen, X_test):
    counter = 0
    for i in range(len(X_test)):
        if collides(X_test[i], chosen):
            counter += 1
    return counter / len(X_test)


def train_inc(x_police, X_test):
    X_test = pre_process_police(X_test)
    k = 11000
    temp = make_matrix_to_police(x_police[:k])
    temp = temp[temp[:, 4].argsort()]
    best = choose30(temp)
    print("result of ", k, "is", testResult(best, X_test))
    return [best, testResult(best, X_test)]


def train_inc_weekend(x_police, X_test):
    X_test = pre_process_police_weekend(X_test)
    k = 900
    temp = make_matrix_to_police_weekend(x_police[:k])
    temp = temp[temp[:, 4].argsort()]
    best = choose30(temp)
    print("result of weekend", k, "is", testResultWeekend(best, X_test))
    return [best, testResult(best, X_test)]


def train_inc_not_weekend(x_police, X_test):
    X_test = pre_process_police_not_weekend(X_test)
    k = 900
    temp = make_matrix_to_police_not_weekend(x_police[:k])
    temp = temp[temp[:, 4].argsort()]
    best = choose30(temp)
    print("result of ", k, "is", testResultNotWeekend(best, X_test))
    return [best, testResult(best, X_test)]


def train_police_pre_process(X, X_test):
    xlong = X['Longitude'].to_numpy()
    xlong = np.reshape(xlong, (-1, 1))
    xlat = X['Latitude'].to_numpy()
    xlat = np.reshape(xlat, (-1, 1))
    x_time_h = X['time_h'].to_numpy().astype(int)
    x_time_h = np.reshape(x_time_h, (-1, 1))
    x_time_m = X['time_m'].to_numpy()
    x_time_m = np.reshape(x_time_m, (-1, 1))
    x_to_police = np.hstack((xlong, xlat, x_time_h, x_time_m))

    xlongt = X_test['Longitude'].to_numpy()
    xlongt = np.reshape(xlongt, (-1, 1))
    xlatt = X_test['Latitude'].to_numpy()
    xlatt = np.reshape(xlatt, (-1, 1))
    x_time_ht = X_test['time_h'].to_numpy().astype(int)
    x_time_ht = np.reshape(x_time_ht, (-1, 1))
    x_time_mt = X_test['time_m'].to_numpy()
    x_time_mt = np.reshape(x_time_mt, (-1, 1))
    x_to_policet = np.hstack((xlongt, xlatt, x_time_ht, x_time_mt))
    train_inc(x_to_police, x_to_policet)


def getXYtime(best, X_train):
    for res in best:
        res[0] = X_train.loc[X_train.index[int(res[3])], 'X Coordinate']
        res[1] = X_train.loc[X_train.index[int(res[3])], 'Y Coordinate']
    best = np.array(best)
    return best[:, :3]


def train_police(X_train, X_test):
    # index = list(range(len(X_train)))
    # # index = np.reshape(index, (-1, 1))
    # X_train["index"] = index
    # resW = train_inc_weekend(X_train,X_test)
    # resNW = train_inc_not_weekend(X_train,X_test)
    res = train_inc(X_train, X_test)
    best = res[0]
    converted = getXYtime(best, X_train)
    print(converted)

