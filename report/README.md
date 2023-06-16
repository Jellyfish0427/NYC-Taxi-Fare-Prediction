# New York City Fare Prediction
## Data Processing
### 1. Remove Missing Data

```
train_df = train_df.drop(train_df.loc[train_df.isnull().any(axis=1)].index, axis=0)
```
### 2. Remove Outliers
#### (1) Remove passengers > 7 and < 1
#### (2) Remove positions outside the New York City
#### (3) Remove fare < 2.5 and > 500
#### (4) Remove positions on the water

### 3. Process Time Features

### 4. Calculate Haversing distance

### 5. Calculate Chebyshiv distance

### 6. Add Airports

### 7. Remove Useless Items

### 8. Observe Correlation between Features and Target


## Model Training

## Conclusions

## Reference
