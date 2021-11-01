import pandas as pd
from DataParser import clean_sorted_hitter
import numpy as np
from sklearn.model_selection import train_test_split
from scikit.linear_model import LinearRegression
from scikit.linear_model import Lasso
from scikit.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,r2_score 
import DataParser.py


def performancelinear():
    #first attempt approach with hitter data/stats, then depending on efficacy tweak & add defensive stats
    # NOTE -> hitting stats include (walks, pitches taken, average, obp+slug)
    # DOUBLE NOTE-> defensive stats will have to only include fielding (outcomes of fielding are (double play, assist, error, passed ball *catcher specific might disregard for now*)
    # TRIPLE NOTE-> add baserunning as a variable as well(outcomes of baserunning are stealing, caught stealing, picked off)
    # after hitter stats attempt, run again w/ pitcher data
    # as we don't have framing data currently it might be better to include catchers as just typical positional players
    hitter_data = clean_sorted_hitter()
    x = hitter_data.AVG  
    y = hitter_data.OBP
    z = hitter_data.SLG
    a = hitter_data.K 
    b = hitter_data.BB  
    # add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
    x_train, x_test, y_train, y_test, z_train, z_test, a_train, a_test, b_train,b_test = train_test_split(x,y,z,a,b, test_size=.25,random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train,z_train,a_train,b_train)
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

