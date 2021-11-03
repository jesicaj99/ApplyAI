from os import name
import pandas as pd
from DataParser import clean_defensive_players, clean_sorted_fielding, clean_sorted_hitter, clean_sorted_pitcher, clean_war, clean_warp_hitter, clean_warp_pitcher
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
    x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test = data_preparation()
    regressor = LinearRegression()
    regressor.fit(x_warp_train, y_warp_train)
    y_warp_pred = regressor.predict(x_warp_test)
    print("R2Linear - hitter WARP correlation: " + r2_score(y_warp_test,y_warp_pred) + ", RMSELinear - hitter WARP correlation: " + mean_squared_error(y_warp_test,y_warp_pred))
    regressor.fit(x_war_train, y_war_train)
    y_war_pred = regressor.predict(x_war_test)
    print("R2Linear - hitter WAR correlation: " + r2_score(y_war_test,y_war_pred) + ", RMSELinear - hitter WAR correlation: " + mean_squared_error(y_war_test,y_war_pred))
    regressor.fit(a_warp_train, b_warp_train)
    b_warp_pred = regressor.predict(a_warp_test)
    print("R2Linear - pitcher correlation: " + r2_score(b_warp_test,b_warp_pred) + ", RMSELinear - pitcher correlation: " + mean_squared_error(b_warp_test,b_warp_pred))
    regressor.fit(a_war_train, b_war_train)
    b_war_pred = regressor.predict(a_war_test)
    print("R2Linear - pitcher correlation: " + r2_score(b_war_test,b_war_pred) + ", RMSELinear - pitcher correlation: " + mean_squared_error(b_war_test,b_war_pred))
    figure  = px.scatter(y_warp_pred,y_warp_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WARP)', hover_name = y_warp_test-y_warp_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(y_war_pred,y_war_test, x = 'Predicted Performance Hitter(Linear)', y = 'Actual Performance Hitter(WAR)', hover_name = y_war_test-y_war_pred, title = 'Actual Performance Hitter vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(b_warp_pred,b_warp_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WARP)', hover_name = b_warp_test-b_warp_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
    figure.show()
    figure  = px.scatter(b_war_pred,b_war_test, x = 'Predicted Performance Pitcher(Linear)', y = 'Actual Performance Pitcher(WAR)', hover_name = b_war_test-b_war_pred, title = 'Actual Performance Pitcher vs. Pred. Performance')
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
    defensive_values = clean_defensive_players()  
    war_values = clean_war()

    #defensive_data = clean_sorted_fielding()
    #combined_batter_data = [hitter_data,defensive_data]
    #combined_data = pd.concat(combined_batter_data) 
    #x = combined_data[['AVG', 'K','BB','OBP','SLG','']]  
    x_warp = []
    y_warp = []
    x_war = []
    y_war = []
    #multiple linear regression where x is composed of multiple variables
    for row in hitter_pred_data.iterrows():
        name = row[0]
        if hitter_data.loc[hitter_data['name']==name]:
            x_warp += hitter_data[name,['K','BB','AVG','OBP','SLG']]
            x_war += hitter_data[name,['K','BB','AVG','OBP','SLG']]
            # y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
            y_warp += hitter_pred_data['WARP']
            y_war+=war_values['Total War']
        if defensive_values.get_val(name) != "No record found":
            x_war+= defensive_values.get_val(name)

    # add additional factors based off of rows in the relevant cavs, (player.csv for players, pitcher.csv for pitchers)
    x_warp_train, x_warp_test, y_warp_train, y_warp_test = train_test_split(x_warp,y_warp, test_size=.25,random_state=1)
    x_war_train, x_war_test, y_war_train, y_war_test = train_test_split(x_war, y_war, test_size = .25, random_state=1)
    a_warp = []
    b_warp = []
    a_war=[]
    b_war = []
    for row in pitcher_pred_data.iterrows():
        name = row[0]
        if pitcher_data.loc[pitcher_data['name']==name]:
            a_warp += pitcher_data[name,['IP', 'BB','K','HR','ERA']]
            # y is pulled from a separate database that I pulled to actually get the x variables to predict a "performance number" rather than a correlation between two statistics 
            b_warp+= pitcher_pred_data['WARP']
            b_war+=war_values['Primary WAR']
    a_warp_train, a_warp_test, b_warp_train, b_warp_test = train_test_split(a_warp,b_warp, test_size=.25,random_state=1)
    a_war_train, a_war_test, b_war_train, b_war_test = train_test_split(a_war,b_war, test_size=.25,random_state=1)
    

    return (x_warp_train,x_warp_test,y_warp_train,y_warp_test,x_war_train,x_war_test,y_war_train,y_war_test,a_warp_train,a_warp_test,b_warp_train,b_warp_test,a_war_train, a_war_test, b_war_train, b_war_test)


def main():
    performancelinear()
    performancelasso()
    performanceelasticnet()
    performanceknnvisualization()

if __name__ == "__main__":
    main()

