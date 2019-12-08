# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

from datetime import datetime
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from backpropagation import BackPropagation
np.set_printoptions(threshold=sys.maxsize)

baseTime = datetime.strptime('2012', '%Y')
nominalCategorical = ["TIPO_PRECIPITACION",
                      "INTENSIDAD_PRECIPITACION", "ESTADO_CARRETERA"]
ordinalCategorical = ["WEEKDAY", "CARRIL_CIRCULACION", "NUMERO_EJES"]
categoricals = nominalCategorical.copy()
categoricals.extend(ordinalCategorical)
yField = "ACCIDENTE"


def eraseEmptyValues(data, inplace=True):
    filterValue = ' '

    for column in data.columns:
        data.drop(data[data[column] == filterValue].index, inplace=inplace)

    return data


def formatOrdinalCategorical(col):
    order = np.unique(col)
    order.sort()
    converted = np.array(col.copy())
    for idx, unique in enumerate(order):
        converted[np.where(converted == unique)] = idx

    return pd.Series(data=converted, name=col.name)


def formatToMonthAndDay(date):
    time = datetime.strptime(date, '%Y-%m-%d')
    return ((time-baseTime).days, time.weekday())


def formatDateToSeconds(date):
    time = date.split('+')[0]
    time = datetime.strptime(
        time, '%H:%M:%S,%f') if ',' in time else datetime.strptime(time, '%H:%M:%S')
    return ((time-baseTime).seconds)


def standarize(data):
    mean = np.mean(data)
    std = np.std(data)
    return pd.Series(data=(np.array(data) - mean) / std, name=data.name)


def formatDataColumns(data):
    dataFrame = pd.DataFrame(data=range(0, data.shape[0]))

    for column in data.columns:
        if column in nominalCategorical:
            dataFrame = dataFrame.join(pd.get_dummies(data[column]))
        elif column in ordinalCategorical:
            values = formatOrdinalCategorical(data[column])
            values = standarize(values)
            dataFrame = dataFrame.join(pd.Series(values))
        else:
            values = standarize(data[column])
            dataFrame = dataFrame.join(pd.Series(values))

    dataFrame.pop(0)
    return dataFrame


def getFECHA_HORAformated(data):
    (days, weekday, timeInSeconds) = formatFechaHora(data.pop('FECHA_HORA'))
    timeDataFrame = pd.DataFrame({"DAYS": days,
                                  "WEEKDAY": weekday,
                                  "timeInSeconds": timeInSeconds})

    data = data.join(timeDataFrame)

    return data


def train_test_split(X, Y, test_size=0.3):
    pass


def formatFechaHora(fecha_hora):
    timeInSeconds = []
    month = []
    weekday = []

    for time in fecha_hora:
        time = time.split(' ')
        (m, w) = formatToMonthAndDay(time[0])
        t = formatDateToSeconds(time[1])
        month.append(m)
        weekday.append(w)
        timeInSeconds.append(t)

    return (month, weekday, timeInSeconds)


def getRandomIndexChoice(length, test_size=0.3):
    testChoice = np.random.choice(length, int(length*test_size), replace=False)
    trainChoice = np.delete(np.array(np.arange(0, length)), testChoice)
    return (trainChoice, testChoice)


def initValues():
    data = pd.read_excel("Datos_PrActica_1_BPNN.xls")

    eraseEmptyValues(data, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    
    data = getFECHA_HORAformated(data)
    (trainIndex, testIndex) = getRandomIndexChoice(len(data))
    

    Y = np.array(
        [1 if 'Yes' == value else 0 for value in data.pop("ACCIDENTE")])

    X = np.array(formatDataColumns(data))
    
    x_train, x_test, y_train, y_test = (X[trainIndex, :],
                                        X[testIndex, :],
                                        Y[trainIndex].reshape(-1, 1),
                                        Y[testIndex].reshape(-1, 1))
    
    NN = BackPropagation()
    NN.load_model('m.json')
    NN.load_weights('w.json')
    NN.fit(x_train, y_train, x_test, y_test, 2, [8, 4])
    NN.save_weights('w.json')
    NN.save_model('m.json')

if __name__ == "__main__":
    initValues()
