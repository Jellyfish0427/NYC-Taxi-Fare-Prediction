# New York City Fare Prediction
## Flow Chart
![flow chart](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/26540318-d7e1-405d-90f2-7af11281179d)

## Data Processing
### 1. Remove Missing Data
<img width="187" alt="截圖 2023-06-16 下午3 45 28" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/2f6633e5-a950-4104-99bf-a4c04eab4627">   

### 2. Remove Outliers
#### (1) Remove passengers > 7 and < 1
![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/dab61e48-5cf5-4015-9f40-9feb21856786)  

#### (2) Remove fare < 2.5 and > 500
![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/a20fc199-4cc3-4018-932c-af1690c52149)  

#### (3) Remove positions outside the New York City
<img width="496" alt="截圖 2023-06-16 下午3 47 36" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/4c46f73c-ce94-45d2-89c4-f1b79bf9f07a">   

- Longitude: -72.8 ~ -74.5
- Latitude: 40.5 ~ 41.8

#### (4) Remove positions on the water
The mask of water:

<img width="441" alt="截圖 2023-06-16 下午3 50 37" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/424292c4-f09e-4f7b-a8a8-eebcd7d4c6ad">

Remove positions on water:   
<img width="696" alt="截圖 2023-06-16 下午3 51 50" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/9fb92fee-2a9f-4021-8964-db88b4fdce16">

### 3. Process Time Features
Split ‘pickup_datatime’ into new features:  
(1) year   
(2) month  
(3) weekday  
(4) hour  

### 4. Calculate Haversine distance
The Haversine distance is the angular distance between two points on the surface of a sphere, it can be calculated with given longitude and latitude.  

#### Haversine formula:  
<img width="307" alt="截圖 2023-06-16 下午5 01 29" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/b6643f25-a35e-40fa-a26d-578229afe236">     

where φ is latitude, λ is longitude, R is earth's radius (mean radius = 6,371km)   

![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/7fd2ce8c-bdf7-4f66-9a85-2231183dc5b1)  

### 5. Calculate Chebyshiv distance
Since New York City is built in a grid plan, Chebyshev distance can be used to represent the distance between 2 locations in New York City.  
#### Chebyshiv distance:  
![截圖 2023-06-16 下午5 49 37](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/b99da61b-11c7-4bc8-b469-a528182a8db2)  

![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/9e235f51-52c4-4294-b5a1-2674f93768f7)

### 6. Calculate Haversine bearing



### 7. Add Airports
(1) JFK Airport: Longitude -73.7781, Latitude 40.6413  
(2) LGA Airport: Longitude -73.8740, Latitude 40.7769  
(3) EWR Airport: Longitude -74.1745, Latitude 40.6895  

![截圖 2023-06-16 下午4 25 12](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/a43d54f3-2b30-478e-bb1b-73a96a4fd6d3)  

### 8. Remove Useless Items
Remove ‘key’ and 'pickup_datatime’.

### 9. Observe Correlation between Features and Target
![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/a1bff4d9-2f99-4fa0-aa2c-00b4eea6c828)   


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

## Conclusions
