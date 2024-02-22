# New York City Fare Prediction
## Flow Chart
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/flow_chart.png" width="600">

## Data Processing
### 1. Remove Missing Data
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/missing_data.png" width="200">
  

### 2. Remove Outliers
#### (1) Remove passengers > 7 and < 1
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/passengers.png" width="800">

#### (2) Remove fare < 2.5 and > 500
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/fare.png" width="800"> 

#### (3) Remove positions outside the New York City
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/nyc_map.png" width="400">

- Longitude: -72.8 ~ -74.5
- Latitude: 40.5 ~ 41.8

#### (4) Remove positions on the water
The mask of water:  
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/mask_of_water.png" width="400">

Remove positions on water:   
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/remove_on_water.png" width="1000">

### 3. Process Time Features
Split ‘pickup_datatime’ into new features:  
- year   
- month  
- weekday  
- hour  

### 4. Calculate Haversine distance
The Haversine distance is the angular distance between two points on the surface of a sphere, it can be calculated with given longitude and latitude.  

#### Haversine formula:  
$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos \phi_1 \cdot \cos \phi_2 \cdot \sin^2\left(\frac{\Delta\lambda}{2}\right)$  

$c = 2 \cdot \text{atan2}\left( \sqrt{a}, \sqrt{1−a} \right)$  

$d = R \cdot c$  

where φ is latitude, λ is longitude, R is earth's radius (mean radius = 6,371km)   

<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/Haversine.png" width="200">

### 5. Calculate Chebyshiv distance
Since New York City is built in a grid plan, Chebyshev distance can be used to represent the distance between 2 locations in New York City.  
#### Chebyshiv distance:  
$D_{Chebyshev}(x, y) = \max(|x_i - y_i|)$   

<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/Chebyshiv.png" width="200">


### 6. Calculate Haversine bearing

### 7. Calculate Manhatten distance

### 8. Add Airports
(1) JFK Airport: Longitude -73.7781, Latitude 40.6413  
(2) LGA Airport: Longitude -73.8740, Latitude 40.7769  
(3) EWR Airport: Longitude -74.1745, Latitude 40.6895  

<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/airports.png" width="400">


### 9. Remove Useless Items
Remove ‘key’ and 'pickup_datatime’.

### 10. Observe Correlation between Features and Target
<img src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/blob/main/img/heatmap.png" width="1000">
  
We can see that the distance features have the greatest correlation with fare.    

## Model Training
### 1. XGBoost
```js
params = {
    'max_depth': 8,
    'n_estimators':500,
    'gamma' :0.,
    'eta':.025, 
    'subsample': 1.0,
    'colsample_bytree': 0.8, 
    'objective':'reg:linear',
    'eval_metric':'rmse',
    'silent': 0,
    'verbosity' : 0,
    'random_state' : 42,
    'tree_method' : 'gpu_hist' #Use GPU
}
```
```js
def XGBmodel(X_train,X_valid,y_train,y_valid,params):
    matrix_train = xgb.DMatrix(X_train,label=y_train)
    matrix_valid = xgb.DMatrix(X_valid,label=y_valid)
    model=xgb.train(params=params,
                    dtrain=matrix_train,num_boost_round=10000, 
                    early_stopping_rounds=100,evals=[(matrix_valid,'valid')])
    return model

model = XGBmodel(X_train,X_valid,y_train,y_valid,params)
```  
Score: 2.95230  

### 2. LightGBM
```js
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
    }
```
```js
def LGBmodel(params,X_train,y_train,X_valid,y_valid):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    #model = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=lgb_eval,early_stopping_rounds=50)
    model = lgb.train(params, lgb_train, num_boost_round=50000, valid_sets=lgb_eval,early_stopping_rounds=500, verbose_eval=500)
    return model

model = LGBmodel(params,X_train,y_train,X_valid,y_valid)
```
Score: 2.96653   

### 3. Ensemble
XGBoost * 0.1 + LightGBM * 0.2 + XGBoost(without Haversine bearing)*0.3 + LightGBM(without Haversine bearing)*0.4   
Score: 2.87272  

*the models were trained in dataset 15M
 
## Future Direction
### 1. Increasing the number of samples in the training set
We have found in our experiments that increasing the number of samples in the training set helps improve the accuracy of model predictions. However, it comes at the cost of consuming more hardware resources and time. In our future work, we plan to increase the sample size and the number of iterations for each model in order to effectively reduce the root mean square error (RMSE).

### 2. Trying different models for training
In this competition, we referred to the works of other participants and noticed that some of them did not use linear regression models. In our feature correlation analysis, we also observed that some features had low correlation with the target price. However, when we removed these features and trained the model, the predictive performance was not satisfactory. Therefore, we plan to incorporate some non-linear regression models into our ensemble model to achieve better results.

### 3. Identifying other key factors affecting price
In this competition, we considered various factors that could potentially influence the price. In addition to the features we ultimately used, we also explored incorporating seasonal or temporal information as feature categories. However, the results were not promising. Moving forward, we plan to explore the inclusion of weather-related factors in our analysis, with the expectation of achieving better results.


## Conclusion
In this work, we mostly put our effort on data pre-processing. In the beginning, besides features given in training data, we only considered calculating Haversine distance between pickup and drop-off location intuitively. However, the outcome was not satisfactory. Consequently, we increased diversity of training features by calculating different types of distance as well as adding popular landmarks in the city and attained better
result. In data processing, it is important to consider all possible factors that may influence the occurrence of events. The other point is that increasing the number of training samples improves the accuracy of models’ predictions. Through a gradual expansion of the training dataset, we obtained prediction results with enhanced precision.




