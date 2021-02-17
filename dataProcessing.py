import pandas as pd

def populateList():
    sti = "https://www.basketball-reference.com/leagues/NBA_20{}_per_game.html"
    urlList = []
    for i in range(21):
        year = str(i)
        if i < 10:
            year = year.zfill(2)
        url = sti.format(year)
        urlList.append(url)
    return urlList

def cleanMultPlayer(df):
    tempTable = df
    pastplayer = ''
    for index, row in tempTable.iterrows():
        curplayer = row['Player']
        if curplayer == pastplayer:
            tempTable.drop([index], inplace=True)
        pastplayer = curplayer
    return tempTable

def combineTwoList(df1, df2):
    comb = pd.merge(df1, df2, on='Player')
    comb = comb.fillna(0)
    return comb

def generateData(urlList):
    comb = pd.DataFrame()
    for i in range(len(urlList) - 1):
        d1 = pd.read_html(urlList[i], header=0)
        d2 = pd.read_html(urlList[i+1], header=0)
        df1 = d1[0]
        df2 = d2[0]
        df1 = df1.drop(df1[df1.Age == 'Age'].index)
        df2 = df2.drop(df2[df2.Age == 'Age'].index)
        df1 = cleanMultPlayer(df1)
        df2 = cleanMultPlayer(df2)
        temp = combineTwoList(df1, df2)
        comb = comb.append(temp, ignore_index=True)
    comb.to_excel("video.xlsx")
    return comb

def dataProcessing():
    urlList = populateList()
    com = generateData(urlList)
    return com

