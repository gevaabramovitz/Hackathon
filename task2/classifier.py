import sys
import pickle

from datetime import datetime
import numpy as np
import pandas as pd

PKL_FILE = "pickle_model.pkl"

NOT_RELEVANT_FE = ["ID", "Location","X Coordinate", "Y Coordinate",
                   "Updated On", "Year", "FBI Code", "IUCR",
                   "Case Number", "Description", "District"]
crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE',
               3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_dict_reverse = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
                       'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}
true_false_dict = {"FALSE": 0, "TRUE": 1}

HOLIDAYS = {1: [1, 2], 2: [], 3: [], 4: [], \
            5: [], 6: [], 7: [4], 8: [], 9: [1, 2, 3, 4, 5, 6, 7], 10: [],
            11: [22, 23, 24, 25, 26, 27, 28],
            12: [24, 25, 26]}

FEATURES = ['Arrest', 'Domestic', 'Community Area', 'Latitude', 'Longitude', 'weekday 0', 'weekday 1',
            'weekday 2', 'weekday 3', 'weekday 4', 'weekday 5', 'weekday 6',
            'quarter hour 0', 'quarter hour 1', 'quarter hour 2',
            'quarter hour 3', 'quarter hour 4', 'quarter hour 5',
            'quarter hour 6', 'quarter hour 7', 'quarter hour 8',
            'quarter hour 9', 'quarter hour 10', 'quarter hour 11',
            'quarter hour 12', 'quarter hour 13', 'quarter hour 14',
            'quarter hour 15', 'quarter hour 16', 'quarter hour 17',
            'quarter hour 18', 'quarter hour 19', 'quarter hour 20',
            'quarter hour 21', 'quarter hour 22', 'quarter hour 23',
            'quarter hour 24', 'quarter hour 25', 'quarter hour 26',
            'quarter hour 27', 'quarter hour 28', 'quarter hour 29',
            'quarter hour 30', 'quarter hour 31', 'quarter hour 32',
            'quarter hour 33', 'quarter hour 34', 'quarter hour 35',
            'quarter hour 36', 'quarter hour 37', 'quarter hour 38',
            'quarter hour 39', 'quarter hour 40', 'quarter hour 41',
            'quarter hour 42', 'quarter hour 43', 'quarter hour 44',
            'quarter hour 45', 'quarter hour 46', 'quarter hour 47',
            'quarter hour 48', 'quarter hour 49', 'quarter hour 50',
            'quarter hour 51', 'quarter hour 52', 'quarter hour 53',
            'quarter hour 54', 'quarter hour 55', 'quarter hour 56',
            'quarter hour 57', 'quarter hour 58', 'quarter hour 59',
            'quarter hour 60', 'quarter hour 61', 'quarter hour 62',
            'quarter hour 63', 'quarter hour 64', 'quarter hour 65',
            'quarter hour 66', 'quarter hour 67', 'quarter hour 68',
            'quarter hour 69', 'quarter hour 70', 'quarter hour 71',
            'quarter hour 72', 'quarter hour 73', 'quarter hour 74',
            'quarter hour 75', 'quarter hour 76', 'quarter hour 77',
            'quarter hour 78', 'quarter hour 79', 'quarter hour 80',
            'quarter hour 81', 'quarter hour 82', 'quarter hour 83',
            'quarter hour 84', 'quarter hour 85', 'quarter hour 86',
            'quarter hour 87', 'quarter hour 88', 'quarter hour 89',
            'quarter hour 90', 'quarter hour 91', 'quarter hour 92',
            'quarter hour 93', 'quarter hour 94', 'quarter hour 95', 'time',
            'is_holiday', 'Beat: 111', 'Beat: 112', 'Beat: 113', 'Beat: 114',
            'Beat: 121', 'Beat: 122', 'Beat: 123', 'Beat: 124', 'Beat: 131',
            'Beat: 132', 'Beat: 133', 'Beat: 211', 'Beat: 212', 'Beat: 213',
            'Beat: 214', 'Beat: 215', 'Beat: 221', 'Beat: 222', 'Beat: 223',
            'Beat: 224', 'Beat: 225', 'Beat: 231', 'Beat: 232', 'Beat: 233',
            'Beat: 234', 'Beat: 235', 'Beat: 311', 'Beat: 312', 'Beat: 313',
            'Beat: 314', 'Beat: 321', 'Beat: 322', 'Beat: 323', 'Beat: 324',
            'Beat: 331', 'Beat: 332', 'Beat: 333', 'Beat: 334', 'Beat: 411',
            'Beat: 412', 'Beat: 413', 'Beat: 414', 'Beat: 421', 'Beat: 422',
            'Beat: 423', 'Beat: 424', 'Beat: 431', 'Beat: 432', 'Beat: 433',
            'Beat: 434', 'Beat: 511', 'Beat: 512', 'Beat: 513', 'Beat: 522',
            'Beat: 523', 'Beat: 524', 'Beat: 531', 'Beat: 532', 'Beat: 533',
            'Beat: 611', 'Beat: 612', 'Beat: 613', 'Beat: 614', 'Beat: 621',
            'Beat: 622', 'Beat: 623', 'Beat: 624', 'Beat: 631', 'Beat: 632',
            'Beat: 633', 'Beat: 634', 'Beat: 711', 'Beat: 712', 'Beat: 713',
            'Beat: 714', 'Beat: 715', 'Beat: 722', 'Beat: 723', 'Beat: 724',
            'Beat: 725', 'Beat: 726', 'Beat: 731', 'Beat: 732', 'Beat: 733',
            'Beat: 734', 'Beat: 735', 'Beat: 811', 'Beat: 812', 'Beat: 813',
            'Beat: 814', 'Beat: 815', 'Beat: 821', 'Beat: 822', 'Beat: 823',
            'Beat: 824', 'Beat: 825', 'Beat: 831', 'Beat: 832', 'Beat: 833',
            'Beat: 834', 'Beat: 835', 'Beat: 911', 'Beat: 912', 'Beat: 913',
            'Beat: 914', 'Beat: 915', 'Beat: 921', 'Beat: 922', 'Beat: 923',
            'Beat: 924', 'Beat: 925', 'Beat: 931', 'Beat: 932', 'Beat: 933',
            'Beat: 934', 'Beat: 935', 'Beat: 1011', 'Beat: 1012', 'Beat: 1013',
            'Beat: 1014', 'Beat: 1021', 'Beat: 1022', 'Beat: 1023',
            'Beat: 1024', 'Beat: 1031', 'Beat: 1032', 'Beat: 1033',
            'Beat: 1034', 'Beat: 1111', 'Beat: 1112', 'Beat: 1113',
            'Beat: 1114', 'Beat: 1115', 'Beat: 1121', 'Beat: 1122',
            'Beat: 1123', 'Beat: 1124', 'Beat: 1125', 'Beat: 1131',
            'Beat: 1132', 'Beat: 1133', 'Beat: 1134', 'Beat: 1135',
            'Beat: 1211', 'Beat: 1212', 'Beat: 1213', 'Beat: 1214',
            'Beat: 1215', 'Beat: 1221', 'Beat: 1222', 'Beat: 1223',
            'Beat: 1224', 'Beat: 1225', 'Beat: 1231', 'Beat: 1232',
            'Beat: 1233', 'Beat: 1234', 'Beat: 1235', 'Beat: 1411',
            'Beat: 1412', 'Beat: 1413', 'Beat: 1414', 'Beat: 1421',
            'Beat: 1422', 'Beat: 1423', 'Beat: 1424', 'Beat: 1431',
            'Beat: 1432', 'Beat: 1433', 'Beat: 1434', 'Beat: 1511',
            'Beat: 1512', 'Beat: 1513', 'Beat: 1522', 'Beat: 1523',
            'Beat: 1524', 'Beat: 1531', 'Beat: 1532', 'Beat: 1533',
            'Beat: 1611', 'Beat: 1612', 'Beat: 1613', 'Beat: 1614',
            'Beat: 1621', 'Beat: 1622', 'Beat: 1623', 'Beat: 1624',
            'Beat: 1631', 'Beat: 1632', 'Beat: 1633', 'Beat: 1634',
            'Beat: 1651', 'Beat: 1652', 'Beat: 1653', 'Beat: 1654',
            'Beat: 1655', 'Beat: 1711', 'Beat: 1712', 'Beat: 1713',
            'Beat: 1722', 'Beat: 1723', 'Beat: 1724', 'Beat: 1731',
            'Beat: 1732', 'Beat: 1733', 'Beat: 1811', 'Beat: 1812',
            'Beat: 1813', 'Beat: 1814', 'Beat: 1821', 'Beat: 1822',
            'Beat: 1823', 'Beat: 1824', 'Beat: 1831', 'Beat: 1832',
            'Beat: 1833', 'Beat: 1834', 'Beat: 1911', 'Beat: 1912',
            'Beat: 1913', 'Beat: 1914', 'Beat: 1915', 'Beat: 1921',
            'Beat: 1922', 'Beat: 1923', 'Beat: 1924', 'Beat: 1925',
            'Beat: 1931', 'Beat: 1932', 'Beat: 1933', 'Beat: 1934',
            'Beat: 1935', 'Beat: 2011', 'Beat: 2012', 'Beat: 2013',
            'Beat: 2022', 'Beat: 2023', 'Beat: 2024', 'Beat: 2031',
            'Beat: 2032', 'Beat: 2033', 'Beat: 2211', 'Beat: 2212',
            'Beat: 2213', 'Beat: 2221', 'Beat: 2222', 'Beat: 2223',
            'Beat: 2232', 'Beat: 2233', 'Beat: 2234', 'Beat: 2411',
            'Beat: 2412', 'Beat: 2413', 'Beat: 2422', 'Beat: 2423',
            'Beat: 2424', 'Beat: 2431', 'Beat: 2432', 'Beat: 2433',
            'Beat: 2511', 'Beat: 2512', 'Beat: 2513', 'Beat: 2514',
            'Beat: 2515', 'Beat: 2521', 'Beat: 2522', 'Beat: 2523',
            'Beat: 2524', 'Beat: 2525', 'Beat: 2531', 'Beat: 2532',
            'Beat: 2533', 'Beat: 2534', 'Beat: 2535', 'ABANDONED BUILDING',
            'AIRCRAFT', 'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA',
            'AIRPORT BUILDING NON-TERMINAL - SECURE AREA',
            'AIRPORT EXTERIOR - NON-SECURE AREA',
            'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT PARKING LOT',
            'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA',
            'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
            'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA',
            'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA',
            'AIRPORT TRANSPORTATION SYSTEM (ATS)',
            'AIRPORT VENDING ESTABLISHMENT', 'ALLEY', 'ANIMAL HOSPITAL',
            'APARTMENT', 'APPLIANCE STORE', 'ATHLETIC CLUB',
            'ATM (AUTOMATIC TELLER MACHINE)', 'AUTO / BOAT / RV DEALERSHIP',
            'BANK', 'BAR OR TAVERN', 'BARBERSHOP', 'BOAT / WATERCRAFT',
            'BOWLING ALLEY', 'BRIDGE', 'CAR WASH', 'CEMETARY', 'CHA APARTMENT',
            'CHA HALLWAY / STAIRWELL / ELEVATOR', 'CHA PARKING LOT / GROUNDS',
            'CHURCH / SYNAGOGUE / PLACE OF WORSHIP', 'CLEANING STORE',
            'COIN OPERATED MACHINE', 'COLLEGE / UNIVERSITY - GROUNDS',
            'COLLEGE / UNIVERSITY - RESIDENCE HALL',
            'COMMERCIAL / BUSINESS OFFICE', 'CONSTRUCTION SITE',
            'CONVENIENCE STORE', 'CREDIT UNION', 'CTA BUS', 'CTA BUS STOP',
            'CTA PARKING LOT / GARAGE / OTHER PROPERTY', 'CTA PLATFORM',
            'CTA STATION', 'CTA TRACKS - RIGHT OF WAY', 'CTA TRAIN',
            'CURRENCY EXCHANGE', 'DAY CARE CENTER', 'DEPARTMENT STORE',
            'DRIVEWAY - RESIDENTIAL', 'DRUG STORE',
            'FACTORY / MANUFACTURING BUILDING', 'FEDERAL BUILDING',
            'FIRE STATION', 'FOREST PRESERVE', 'GAS STATION',
            'GOVERNMENT BUILDING / PROPERTY', 'GROCERY FOOD STORE',
            'HIGHWAY / EXPRESSWAY', 'HOSPITAL BUILDING / GROUNDS',
            'HOTEL / MOTEL', 'JAIL / LOCK-UP FACILITY', 'KENNEL',
            'LAKEFRONT / WATERFRONT / RIVERBANK', 'LIBRARY',
            'MEDICAL / DENTAL OFFICE', 'MOVIE HOUSE / THEATER', 'NEWSSTAND',
            'NURSING / RETIREMENT HOME', 'OTHER (SPECIFY)',
            'OTHER COMMERCIAL TRANSPORTATION',
            'OTHER RAILROAD PROPERTY / TRAIN DEPOT', 'PARK PROPERTY',
            'PARKING LOT / GARAGE (NON RESIDENTIAL)', 'PAWN SHOP',
            'POLICE FACILITY / VEHICLE PARKING LOT', 'POOL ROOM', 'RESIDENCE',
            'RESIDENCE - GARAGE', 'RESIDENCE - PORCH / HALLWAY',
            'RESIDENCE - YARD (FRONT / BACK)', 'RESTAURANT',
            'SCHOOL - PRIVATE BUILDING', 'SCHOOL - PRIVATE GROUNDS',
            'SCHOOL - PUBLIC BUILDING', 'SCHOOL - PUBLIC GROUNDS', 'SIDEWALK',
            'SMALL RETAIL STORE', 'SPORTS ARENA / STADIUM', 'STREET',
            'TAVERN / LIQUOR STORE', 'TAXICAB', 'VACANT LOT / LAND',
            'VEHICLE - COMMERCIAL',
            'VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS',
            'VEHICLE - COMMERCIAL: TROLLEY BUS', 'VEHICLE - DELIVERY TRUCK',
            'VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)',
            'VEHICLE NON-COMMERCIAL', 'WAREHOUSE', 'Ward: 1', 'Ward: 2',
            'Ward: 3', 'Ward: 4', 'Ward: 5', 'Ward: 6', 'Ward: 7', 'Ward: 8',
            'Ward: 9', 'Ward: 10', 'Ward: 11', 'Ward: 12', 'Ward: 13',
            'Ward: 14', 'Ward: 15', 'Ward: 16', 'Ward: 17', 'Ward: 18',
            'Ward: 19', 'Ward: 20', 'Ward: 21', 'Ward: 22', 'Ward: 23',
            'Ward: 24', 'Ward: 25', 'Ward: 26', 'Ward: 27', 'Ward: 28',
            'Ward: 29', 'Ward: 30', 'Ward: 31', 'Ward: 32', 'Ward: 33',
            'Ward: 34', 'Ward: 35', 'Ward: 36', 'Ward: 37', 'Ward: 38',
            'Ward: 39', 'Ward: 40', 'Ward: 41', 'Ward: 42', 'Ward: 43',
            'Ward: 44', 'Ward: 45', 'Ward: 46', 'Ward: 47', 'Ward: 48',
            'Ward: 49', 'Ward: 50']


def get_parsed_time(X):
    """
    converting time in 'Date' feature into time vector.
    :param X: data set.
    :return: X: updated data set. time: column of time.
    """
    times = None
    flag = False
    while X.shape[0] > 0:
        try:
            times = X['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y '
                                                                   '%I:%M:%S %p'))
            break
        except ValueError:
            i = 0
            while i < X.shape[0]:
                try:
                    datetime.strptime(X.loc[X.index[i], 'Date'],
                                      '%m/%d/%Y %I:%M:%S %p')
                    flag = False
                    i += 1
                except ValueError:
                    X.drop(X.index[i], inplace=True)
                    flag = True
                    break
            if not flag:
                break

    return X, times


def remove_redundant_cols(X):
    """
    removing redundant columns.
    :param X: data set
    :return: X: updated data set.
    """
    # column erasing
    X.drop(NOT_RELEVANT_FE, axis=1, inplace=True)
    del X['Block']

    X["Location Description"] = X["Location Description"].fillna(
        "PROB_DECEPTIVE")
    return X


def create_features_related_to_time(X, times):
    """
    creates features: days in a week, chunks of quarter hours.
    :param X: data sets.
    :param times: time feature.
    :return: X: updated data set.
    """
    X['weekday'] = times.apply(lambda x: x.weekday())
    dummies = pd.get_dummies(X['weekday']). \
        rename(columns={i: "weekday " + str(i) for i in range(7)})
    X = pd.concat([X, dummies], axis=1)
    del X['weekday']

    X['half_hours_48'] = times.apply(
        lambda x: (int(x.hour * 4)) + (int(x.strftime('%M')) // 15))
    dummies = pd.get_dummies(X['half_hours_48']). \
        rename(columns={i: "quarter hour " + str(i) for i in range(96)})
    X = pd.concat([X, dummies], axis=1)
    del X['half_hours_48']

    X['time'] = times.apply(lambda x: x.hour * 60 + int(x.strftime('%M')))

    # insert holiday check.
    X["is_holiday"] = times.apply(lambda x: is_holiday(x)).astype(int)
    del X["Date"]
    return X


def create_dummies(X):
    """
    creates dummies variables from Beat, Location Description, and Ward.
    :param X: data set.
    :return: updated data set.
    """
    dummies = pd.get_dummies(X['Beat']).rename(
        columns={i: "Beat: " + str(i) for i in range(len(X))})
    X = pd.concat([X, dummies], axis=1)
    del X['Beat']
    dummies = pd.get_dummies(X['Location Description'])
    X = pd.concat([X, dummies], axis=1)
    del X['Location Description']
    dummies = pd.get_dummies(X['Ward']).rename(
        columns={i: "Ward: " + str(i) for i in range(1, 51)})
    X = pd.concat([X, dummies], axis=1)
    del X['Ward']
    return X


def is_holiday(time_obj):
    """
    checks if the object is holiday.
    :param time_obj: object
    :return: True or False
    """
    return int(time_obj.strftime('%d')) in HOLIDAYS[int(time_obj.strftime(
        '%m'))]


def pre_process_train(X):
    """
    pre-proccessing on the data.
    :param X: data set.
    :return: updated data set.
    """
    X = pd.read_csv(X, index_col=0)
    X = remove_redundant_cols(X)

    # parse time - catch wrong formats
    X, times = get_parsed_time(X)

    X = create_features_related_to_time(X, times)

    # get dummies variables from: Beat, Location Description, Ward, Block:
    X = create_dummies(X)

    X = X.T.reindex(FEATURES).T.fillna(0)

    return X


def predict(X):
    """

    :param X: data set.
    :return: the fitted model.
    """
    X = pre_process_train(X)
    with open(PKL_FILE, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model.predict(X)



def minutes_to_date(date, mins):
    date = date.split(" ")[0]
    pm = False
    time_h = mins // 60
    time_m = mins - (60 * time_h)
    time_m = str(int(time_m)) if time_m > 9 else "0" + str(int(time_m))
    if time_h > 12:
        time_h = str(int(time_h - 12))
        pm = True
    else:
        time_h = str(int(time_h))
        pm = False
    time_h = time_h if int(time_h) > 9 else "0" + time_h

    time = date + " " + time_h + ":" + time_m + ":00"
    return time + " PM" if pm else time + " AM"


def results_to_tuples(date):
    results = [[1.176389e+06, 1.900278e+06, 9.300000e+02],
               [1.184252e+06, 1.853187e+06, 9.430000e+02],
               [1.176365e+06, 1.900502e+06, 7.350000e+02],
               [1.176260e+06, 1.904329e+06, 1.225000e+03],
               [1.176517e+06, 1.895340e+06, 6.300000e+02],
               [1.175401e+06, 1.901736e+06, 9.600000e+02],
               [1.150281e+06, 1.898067e+06, 1.088000e+03],
               [1.176468e+06, 1.904201e+06, 8.200000e+02],
               [1.176300e+06, 1.902537e+06, 7.800000e+02],
               [1.160483e+06, 1.861810e+06, 9.000000e+02],
               [1.171721e+06, 1.920469e+06, 1.050000e+03],
               [1.153720e+06, 1.894499e+06, 6.900000e+02],
               [1.195031e+06, 1.853094e+06, 5.490000e+02],
               [1.142012e+06, 1.899172e+06, 6.900000e+02],
               [1.175704e+06, 1.900816e+06, 6.600000e+02],
               [1.173593e+06, 1.904710e+06, 7.200000e+02],
               [1.177003e+06, 1.905748e+06, 7.210000e+02],
               [1.176365e+06, 1.900502e+06, 7.950000e+02],
               [1.176257e+06, 1.904583e+06, 1.155000e+03],
               [1.177003e+06, 1.905748e+06, 9.870000e+02],
               [1.176018e+06, 1.904773e+06, 1.050000e+03],
               [1.171463e+06, 1.920708e+06, 9.150000e+02],
               [1.194519e+06, 1.854817e+06, 1.200000e+02],
               [1.175315e+06, 1.898388e+06, 9.600000e+02],
               [1.142353e+06, 1.900632e+06, 6.300000e+02],
               [1.180736e+06, 1.836592e+06, 8.100000e+02],
               [1.167997e+06, 1.932393e+06, 6.900000e+02],
               [1.176342e+06, 1.904782e+06, 0.000000e+00],
               [1.153304e+06, 1.891654e+06, 5.700000e+02],
               [1.170259e+06, 1.902888e+06, 1.170000e+03]]
    cars = []
    for i in range(30):
        cars.append((results[i][0], results[i][1], minutes_to_date(date, results[i][2])))
    return cars


def predict_police(date):
    return results_to_tuples(date)

def send_police_cars(X):
    return predict_police(X)

