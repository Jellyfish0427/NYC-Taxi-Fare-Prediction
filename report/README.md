# New York City Fare Prediction
## Data Processing
### 1. Remove Missing Data

```
train_df = train_df.drop(train_df.loc[train_df.isnull().any(axis=1)].index, axis=0)
```   
<img width="187" alt="截圖 2023-06-16 下午3 45 28" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/2f6633e5-a950-4104-99bf-a4c04eab4627">   

### 2. Remove Outliers
#### (1) Remove passengers > 7 and < 1
![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/3aedf8da-0b66-4b08-af9b-24e73a74e9eb)  

#### (2) Remove fare < 2.5 and > 500
![image](https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/d7fe96a9-da83-4e61-a33a-b5ecbde4be99) 

#### (3) Remove positions outside the New York City
<img width="496" alt="截圖 2023-06-16 下午3 47 36" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/4c46f73c-ce94-45d2-89c4-f1b79bf9f07a">   

- Longitude: -72.8 ~ -74.5
- Latitude: 40.5 ~ 41.8

#### (4) Remove positions on the water
The mask of water:

<img width="441" alt="截圖 2023-06-16 下午3 50 37" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/424292c4-f09e-4f7b-a8a8-eebcd7d4c6ad">

<img width="696" alt="截圖 2023-06-16 下午3 51 50" src="https://github.com/Jellyfish0427/NYC-Taxi-Fare-Prediction/assets/128220508/9fb92fee-2a9f-4021-8964-db88b4fdce16">



### 3. Process Time Features

### 4. Calculate Haversing distance

### 5. Calculate Chebyshiv distance

### 6. Add Airports

### 7. Remove Useless Items

### 8. Observe Correlation between Features and Target


## Model Training

## Conclusions

## Reference
