import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from dataProcessing import dataProcessing, cleanMultPlayer
from staticModel import staticModel


def proccessData(raw):
    df = pd.DataFrame(columns=['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN', 'nPTS', 'nREB',
                               'nAST', 'nBLK', 'nSTL', 'nTO'])
    for i in range(len(raw) - 1):
        df.loc[i] = [raw.at[i+1, 'PTS'], raw.at[i+1, 'REB'], raw.at[i+1, 'AST'],
                     raw.at[i+1, 'BLK'], raw.at[i+1, 'STL'], raw.at[i+1, 'TO'],
                     raw.at[i+1, 'PF'], raw.at[i+1, 'FG%'], raw.at[i+1, 'FT%'], raw.at[i+1, 'MIN'],
                     raw.at[i, 'PTS'], raw.at[i, 'REB'], raw.at[i, 'AST'],
                     raw.at[i, 'BLK'], raw.at[i, 'STL'], raw.at[i, 'TO']]
    return df


def proccessMoreData(raw, predictions):
    df = pd.DataFrame(columns=['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN', 'nPTS', 'nREB',
                               'nAST', 'nBLK', 'nSTL', 'nTO'])
    for i in range(len(raw - 1)):
        df.loc[i] = [raw.at[i, 'PTS'], raw.at[i, 'REB'], raw.at[i, 'AST'],
                     raw.at[i, 'BLK'], raw.at[i, 'STL'], raw.at[i, 'TO'],
                     raw.at[i, 'PF'], raw.at[i, 'FG%'], raw.at[i, 'FT%'], raw.at[i, 'MIN'],
                     predictions[0][5], predictions[0][0], predictions[0][1],
                     predictions[0][3], predictions[0][2], predictions[0][4]]
    return df



def getData(predictions):
    url = input('Enter Website URL: ')
    d = pd.read_html(url, header=0)
    df1 = d[0]
    df1 = df1[['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN']]
    dt1 = proccessData(df1)
    mdt1 = proccessMoreData(df1, predictions)
    df2 = d[1]
    df2 = df2[['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN']]
    dt2 = proccessData(df2)
    mdt2 = proccessMoreData(df2, predictions)
    df3 = d[2]
    df3 = df3[['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN']]
    dt3 = proccessData(df3)
    mdt3 = proccessMoreData(df3, predictions)
    mdt = mdt1.append(mdt2, ignore_index=True, sort=False)
    mdt = mdt.append(mdt3, ignore_index=True, sort=False)
    dt = dt1.append(dt2, ignore_index=True, sort=False)
    dt = dt.append(dt3, ignore_index=True, sort=False)
    comb = dt.append(mdt, ignore_index=True, sort=False)
    return comb

def test():
    url = input('Enter Website URL: ')
    d = pd.read_html(url, header=0)
    df1 = d[1]
    df1 = df1[['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN']]
    dt1 = proccessData(df1)
    print(dt1)


def machinelearn():
    predictions = staticModel()
    comb = getData(predictions)
    X = np.array(comb[['PTS', 'REB', 'AST', 'BLK', 'STL', 'TO', 'PF', 'FG%', 'FT%', 'MIN']])
    Y = np.array(comb[['nPTS', 'nREB', 'nAST', 'nBLK', 'nSTL', 'nTO']])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    model = linear_model.Lasso(alpha=0.01)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)

    playerStat = comb.loc[0]
    predictList = [[playerStat['PTS'], playerStat['REB'], playerStat['AST'], playerStat['BLK'], playerStat['STL'],
                    playerStat['TO'], playerStat['PF'], playerStat['FG%'], playerStat['FT%'], playerStat['MIN']]]
    predictList = model.predict(predictList)

    print('PTS: ', round(predictList[0][0], 0))
    print('RB: ', round(predictList[0][1], 0))
    print('AST: ', round(predictList[0][2], 0))
    print('STL: ', round(predictList[0][3], 0))
    print('BLK: ', round(predictList[0][4], 0))
    print('TOV: ', round(predictList[0][5], 0))

def dynamicModel():
    machinelearn()