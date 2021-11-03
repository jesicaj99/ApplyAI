from os import name
import pandas as pd
from DataParser import clean_sorted_fielding, clean_sorted_hitter, clean_sorted_pitcher, clean_warp_hitter, clean_warp_pitcher
import numpy as np
from sklearn.model_selection import train_test_split
from scikit.linear_model import LinearRegression
from scikit.linear_model import Lasso
from scikit.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error,r2_score, mean_squared_error
import DataParser.py
import plotly.express as px


def performancelinear():
    #first attempt approach with hitter data/stats, then depending on efficacy tweak & add defensive stats
    # NOTE -> hitting stats include (walks, pitches taken, average, obp+slug)
    # DOUBLE NOTE-> defensive stats will have to only include fielding (outcomes of fielding are (double play, assist, error, passed ball *catcher specific might disregard for now*)
    # TRIPLE NOTE-> add baserunning as a variable as well(outcomes of baserunning are stealing, caught stealing, picked off)
    # after hitter stats attempt, run again w/ pitcher data
    # as we don't have framing data currently it might be better to include catchers as just typical positional players
    # QUADRUPLE NOTE-> after further data exploration, this is boiling down into total runs scored + total runs saved into a predictor for player performance -> WARP + defensive stat created, compare to WAR
    # QUINTUPLE NOTE-> For pitchers, it boils down to total runs saved through pitching really
    # pulling data for models from the cleaned data from the dataparser program
    x_train,x_test,y_train,y_test,a_train,a_test,b_train,b_test = data_preparation()
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print("R2Linear - hitter correlation: " + r2_score(y_test,y_pred) + ", RMSELinear - hitter correlation: " + mean_squared_error(y_test,y_pred))
    regressor.fit(a_train, b_train)
    b_pred = regressor.predict(a_test)
    print("R2Linear - pitcher correlation: " + r2_score(b_test,b_pred) + ", RMSELinear - pitcher correlation: " + mean_squared_error(b_test,b_pred))
    figure  = px.scatter(y_pred,y_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(b_pred,b_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
    figure.show()  

def performancelasso():
    x_train,x_test,y_train,y_test,a_train,a_test,b_train,b_test = data_preparation()
    regressor = Lasso()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    print("R2Lasso - hitter correlation: " + r2_score(y_test,y_pred) + ", RMSELasso - hitter correlation: " + mean_squared_error(y_test,y_pred))
    regressor.fit(a_train, b_train)
    b_pred = regressor.predict(a_test)
    print("R2Lasso - pitcher correlation: " + r2_score(b_test,b_pred) + ", RMSELasso - pitcher correlation: " + mean_squared_error(b_test,b_pred))
    figure  = px.scatter(y_pred,y_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(b_pred,b_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
    figure.show()

def performanceelasticnet():
    x_train,x_test,y_train,y_test,a_train,a_test,b_train,b_test = data_preparation()
    regressor = ElasticNet()
    regressor.fit(x_train,y_train)
    y_pred = regressor.predict(x_test)
    print("R2Elastic - hitter correlation: " + r2_score(y_test,y_pred) + ", RMSEElastic - hitter correlation: " + mean_squared_error(y_test,y_pred))
    regressor.fit(a_train, b_train)
    b_pred = regressor.predict(a_test)
    print("R2Elastic - pitcher correlation: " + r2_score(b_test,b_pred) + ", RMSEElastic - pitcher correlation: " + mean_squared_error(b_test,b_pred))
    figure  = px.scatter(y_pred,y_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(b_pred,b_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WARP)', hover_name = y_test-y_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
    figure.show()

def performanceknnvisualization():
    x_train,x_test,y_train,y_test,a_train,a_test,b_train,b_test = data_preparation()
    model_hitter = KNeighborsClassifier(n_neighbors = 3)
    model_hitter.fit(x_train,y_train)
    model_pitcher = KNeighborsClassifier(n_neighbors = 3)
    model_pitcher.fit(a_train,b_train)
    print("Accuracy Score - KNN hitters: " + model_hitter.score(x_test,y_test))
    print("Accuracy Score - KNN pitchers: " + model_pitcher.score(a_test,b_test))
   
def data_preparation():
    hitter_data = clean_sorted_hitter()
    hitter_pred_data = clean_warp_hitter()
    pitcher_data = clean_sorted_pitcher()
    pitcher_pred_data = clean_warp_pitcher()  
    #defensive_data = clean_sorted_fielding()
    #combined_batter_data = [hitter_data,defensive_data]
    #combined_data = pd.concat(combined_batter_data) 
    #x = combined_data[['AVG', 'K','BB','OBP','SLG','']]  
    x = []
    y = []
    #multiple linear regression where x is composed of multiple variables
    for row in hitter_pred_data.iterrows():
        name = row[0]
        if hitter_data.loc[hitter_data['name']==name]:
            x += hitter_data[name,['K','BB','AVG','OBP','SLG']]
            # y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
            y += hitter_pred_data['WARP']
    # add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.25,random_state=1)
    a = []
    b = []
    for row in pitcher_pred_data.iterrows():
        name = row[0]
        if pitcher_data.loc[pitcher_data['name']==name]:
            a += pitcher_data[name,['IP', 'BB','K','HR','ERA']]
            # y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
            b+= pitcher_pred_data['WARP']
    a_train, a_test, b_train, b_test = train_test_split(a,b, test_size=.25,random_state=1)
    return (x_train,x_test,y_train,y_test,a_train,a_test,b_train,b_test)


def main():
    performancelinear()
    performancelasso()
    performanceelasticnet()
    performanceknnvisualization()

if __name__ == "__main__":
    main()

