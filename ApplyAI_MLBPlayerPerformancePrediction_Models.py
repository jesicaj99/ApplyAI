import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scikit.linear_model import LinearRegression
from scikit.linear_model import Lasso
from scikit.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier

def performancelinear():
    #first attempt approach with hitter data/stats, then depending on efficacy tweak & add defensive stats
    # after hitter stats attempt, run again w/ pitcher data
    # as we don't have framing data currently it might be better to include catchers as just typical positional players
    hitterdataset = pd.read_csv('')
    x = dataset.iloc[:,:-1].values   #independent variable array 1st factor
    y = dataset.iloc[:,1].values  #dependent variable vector 2nd factor
    # add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1/3,random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

def performancelasso():

def performanceelasticnet():

def performanceKNNvisualization():

def main():

if __name__ == "__main__":
    main()

