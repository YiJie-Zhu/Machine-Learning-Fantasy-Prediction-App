import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from dataProcessing import dataProcessing, cleanMultPlayer

cp = pd.read_html("https://www.basketball-reference.com/leagues/NBA_2020_per_game.html", header=0)
currentPlayer = cp[0]
currentPlayer = currentPlayer.drop(currentPlayer[currentPlayer.Age == 'Age'].index)
currentPlayer = cleanMultPlayer(currentPlayer)


def getPostion(list, position):
    return list[(list == position).any(axis=1)]


def machinelearn(predictList):
    comb = dataProcessing()
    # comb = getPostion(comb, 'PG')
    X = np.array(
        comb[['Age_x', 'MP_x', 'FG_x', 'FGA_x', 'FT_x', 'FTA_x', 'TRB_x', 'AST_x', 'STL_x', 'BLK_x', 'TOV_x', 'PTS_x']])
    # Y = np.array(comb[['PTS_y']])
    Y = np.array(comb[['TRB_y', 'AST_y', 'STL_y', 'BLK_y', 'TOV_y', 'PTS_y']])
    # # comb.to_excel("test2.xlsx")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    model = linear_model.Lasso(alpha=0.01)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)
    predictions = model.predict(predictList)
    return predictions


def staticModel():
    temp = pd.DataFrame
    while temp.empty:
        py = input('Enter player: ')
        temp = currentPlayer.loc[currentPlayer['Player'] == py]
    predictList = np.array(temp[['Age', 'MP', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']])
    predictions = machinelearn(predictList)


    # print('PTS: ', round(predictions[0][5], 1))
    # print('RB: ', round(predictions[0][0], 1))
    # print('AST: ', round(predictions[0][1], 1))
    # print('STL: ', round(predictions[0][2], 1))
    # print('BLK: ', round(predictions[0][3], 1))
    # print('TOV: ', round(predictions[0][4], 1))
    return predictions



