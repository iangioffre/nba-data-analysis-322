# NBA Statistics Analysis
### Ian Gioffre, Benen Dufresne

This project implements a Random Forest Classifier to predict the salary class of NBA athletes given several arguments:  
* year_started: The year the athlete joined the NBA
* height: The height of the athlete
* weight_class: The weight class of the athlete (weight <= 150 + (weight_class * 30))
* position: The position or positions that the athlete plays

The salary classes are as follows:  
1. $\le$ 100,000
1. \lt 1,000,000
1. $\lt$ 5,000,000
1. < 10,000,000
1. \>= 10,000,000

The predictor can be used with the following url:  
http://TODO/predict?year_start=[year_start]&height=[height]&weight_class=[weight_class]&position=[position]  
where the arguments in square brackets need to be filled in without the brackets.