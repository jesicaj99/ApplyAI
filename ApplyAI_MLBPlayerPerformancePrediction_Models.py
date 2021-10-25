import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scikit.linear_model import LinearRegression
from scikit.linear_model import Lasso
from scikit.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,r2_score 


def performancelinear():
    #first attempt approach with hitter data/stats, then depending on efficacy tweak & add defensive stats
    # NOTE -> hitting stats include (walks, pitches taken, average, obp+slug)
    # DOUBLE NOTE-> defensive stats will have to only include fielding (outcomes of fielding are (double play, assist, error, passed ball *catcher specific might disregard for now*)
    # TRIPLE NOTE-> add baserunning as a variable as well(outcomes of baserunning are stealing, caught stealing, picked off)
    # after hitter stats attempt, run again w/ pitcher data
    # as we don't have framing data currently it might be better to include catchers as just typical positional players
    hitterdataset = pd.read_csv('')
    x = hitterdataset.iloc[:,:-1].values   #independent variable array 1st factor
    y = hitterdataset.iloc[:,1].values  #dependent variable vector 2nd factor
    # add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    return (r2_score(y_test,y_pred))

def performancelasso():
    regressor = Lasso()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    return (r2_score(y_test,y_pred))

def performanceelasticnet():
    regressor = ElasticNet()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    return (r2_score(y_test,y_pred))

#def performanceKNNvisualization():

def main():
    #playerperformancedata 
    performancelinear()
    performancelasso()
    performanceelasticnet()
    #performanceKNNvisualization()

if __name__ == "__main__":
    main()

