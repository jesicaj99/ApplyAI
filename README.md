# ApplyAI Project Team 11 (Siddharth Nair, Jesica Jimenez, John Paul Bartsch)

**What is BaseballPerformanceApproximator(BPA)?**
BPA is a project created by the three of us for the ApplyAI program, effectively attempting to utilize Machine Learning to conduct data exploration on the topic of our choosing,
in this case mlb player's performance data. We are attempting to use this performance data to create different models to best predict a cohesive baseball statistic indicating a 
player's performance for the year called WAR.

**How does BPA work?**
BPA works by taking in data from a variety of datasets, found on Kaggle, Stathead via MLB, and Fangraphs to create training and test sets for our models. From there, the models 
are fit using scikit's various models(linear, lasso, elastic net, KNN) and are visualized with plotly to best see if our predictions based off of the stats being used for pitchers 
and batters respectively, are accurate *enough*. (Top right and bottom left quadrants are the quadrants of the graph that'd indicate a good prediction, top left and bottom right 
indicate an underprediction or over prediction respectively).

**Who should use BPA?**
No one! This is purely an exercise by us to see if we can utilize these machine learning tools to complete a goal of our choosing, and to see the accuracy of the models given the
variables being inputted. We will say that if this model ends up being fairly accurate(approx. 10% error), we'll continue to tweak it to becoming a viable product/alternative 
method of calculating WAR.

**What is BPA's goal?**
BPA's goal is to best approximate WARP, then WAR with the chosen data from the datasets being used. As necessary, more or less data will be used depending on the efficacy of the
model in its various iterations. 

**What is the benefit of BPA?**
The benefit of BPA is seeing a different perspective on the factor's that affect baseball player's performance during a regular season, and whether a simpler selection of variables can be used to approximate a complex statistic w/ many variables going into it (some of which are all but impossible to approximate w/o going through game data per player).



