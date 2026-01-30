# Data Assessment – Guardian Mutual Assurance (GMA)

This notebook performs data exploration and assessment for the Claims, Policyholders, and Third-Parties datasets. 

Objectives:
- Understand data structure and types
- Identify missing values
- Explore target variable distribution
- Detect extreme outliers
- Map dataset relationships
- Highlight key data issues



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
claims = pd.read_excel("claims.xlsx")
policyholders = pd.read_excel("policyholders.xlsx")
third_parties = pd.read_excel("third_parties.xlsx")

```

## Initial Exploration

We first inspect the data using `.head()`, `.info()`, and `.describe()` to understand:
- Column names
- Data types
- Summary statistics



```python
# Claims
print("Claims Table Head:")
display(claims.head())
print("Claims Info:")
display(claims.info())
print("Claims Describe:")
display(claims.describe())

# Policyholders
print("Policyholders Table Head:")
display(policyholders.head())
print("Policyholders Info:")
display(policyholders.info())
print("Policyholders Describe:")
display(policyholders.describe())

# Third-Parties
print("Third-Parties Table Head:")
display(third_parties.head())
print("Third-Parties Info:")
display(third_parties.info())
print("Third-Parties Describe:")
display(third_parties.describe())

```

    Claims Table Head:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Claim_ID</th>
      <th>Policy_ID</th>
      <th>Accident_Date</th>
      <th>FNOL_Date</th>
      <th>Claim_Type</th>
      <th>Claim_Complexity</th>
      <th>Fraud_Flag</th>
      <th>Litigation_Flag</th>
      <th>Estimated_Claim_Amount</th>
      <th>Ultimate_Claim_Amount</th>
      <th>Severity_Band</th>
      <th>Settlement_Date</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CLM30000</td>
      <td>POL14506</td>
      <td>2019-12-19</td>
      <td>2019-12-19</td>
      <td>Theft</td>
      <td>Medium</td>
      <td>False</td>
      <td>True</td>
      <td>5243</td>
      <td>2808.0</td>
      <td>Minor</td>
      <td>2020-03-01</td>
      <td>settled</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLM30001</td>
      <td>POL14338</td>
      <td>2018-12-30</td>
      <td>2018-12-31</td>
      <td>Collision</td>
      <td>Low</td>
      <td>False</td>
      <td>False</td>
      <td>3934</td>
      <td>2952.0</td>
      <td>Minor</td>
      <td>2019-03-23</td>
      <td>settled</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CLM30002</td>
      <td>POL13575</td>
      <td>2021-10-19</td>
      <td>2021-10-19</td>
      <td>Other</td>
      <td>Medium</td>
      <td>False</td>
      <td>False</td>
      <td>153631</td>
      <td>156497.0</td>
      <td>Catastrophic</td>
      <td>2022-04-22</td>
      <td>settled</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CLM30003</td>
      <td>POL10138</td>
      <td>2021-06-18</td>
      <td>2021-06-18</td>
      <td>Weather</td>
      <td>Low</td>
      <td>False</td>
      <td>False</td>
      <td>2812</td>
      <td>1450.0</td>
      <td>Minor</td>
      <td>2021-09-13</td>
      <td>settled</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CLM30004</td>
      <td>POL12316</td>
      <td>2021-03-21</td>
      <td>2021-03-24</td>
      <td>Theft</td>
      <td>Low</td>
      <td>False</td>
      <td>False</td>
      <td>5094</td>
      <td>4243.0</td>
      <td>Minor</td>
      <td>2021-05-26</td>
      <td>settled</td>
    </tr>
  </tbody>
</table>
</div>


    Claims Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8000 entries, 0 to 7999
    Data columns (total 13 columns):
     #   Column                  Non-Null Count  Dtype         
    ---  ------                  --------------  -----         
     0   Claim_ID                8000 non-null   object        
     1   Policy_ID               8000 non-null   object        
     2   Accident_Date           8000 non-null   datetime64[ns]
     3   FNOL_Date               8000 non-null   datetime64[ns]
     4   Claim_Type              8000 non-null   object        
     5   Claim_Complexity        8000 non-null   object        
     6   Fraud_Flag              8000 non-null   bool          
     7   Litigation_Flag         8000 non-null   bool          
     8   Estimated_Claim_Amount  8000 non-null   int64         
     9   Ultimate_Claim_Amount   7575 non-null   float64       
     10  Severity_Band           8000 non-null   object        
     11  Settlement_Date         7575 non-null   datetime64[ns]
     12  Status                  8000 non-null   object        
    dtypes: bool(2), datetime64[ns](3), float64(1), int64(1), object(6)
    memory usage: 703.3+ KB
    


    None


    Claims Describe:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accident_Date</th>
      <th>FNOL_Date</th>
      <th>Estimated_Claim_Amount</th>
      <th>Ultimate_Claim_Amount</th>
      <th>Settlement_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8000</td>
      <td>8000</td>
      <td>8.000000e+03</td>
      <td>7.575000e+03</td>
      <td>7575</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2022-04-02 01:52:51.600000</td>
      <td>2022-04-03 20:20:45.600000256</td>
      <td>1.447988e+04</td>
      <td>1.311868e+04</td>
      <td>2022-07-15 14:21:54.534653440</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2018-09-19 00:00:00</td>
      <td>2018-09-19 00:00:00</td>
      <td>5.500000e+02</td>
      <td>3.320000e+02</td>
      <td>2018-10-25 00:00:00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2020-06-20 00:00:00</td>
      <td>2020-06-22 18:00:00</td>
      <td>2.205750e+03</td>
      <td>1.578000e+03</td>
      <td>2020-10-11 12:00:00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2022-04-06 00:00:00</td>
      <td>2022-04-07 00:00:00</td>
      <td>4.416000e+03</td>
      <td>3.409000e+03</td>
      <td>2022-07-14 00:00:00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2024-01-05 00:00:00</td>
      <td>2024-01-06 12:00:00</td>
      <td>1.167475e+04</td>
      <td>9.744000e+03</td>
      <td>2024-04-17 00:00:00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2025-09-17 00:00:00</td>
      <td>2025-09-22 00:00:00</td>
      <td>1.064239e+06</td>
      <td>1.005590e+06</td>
      <td>2027-03-29 00:00:00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.873981e+04</td>
      <td>3.810405e+04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    Policyholders Table Head:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Policy_ID</th>
      <th>Customer_ID</th>
      <th>Age_of_Driver</th>
      <th>Gender</th>
      <th>Occupation</th>
      <th>Region</th>
      <th>Annual_Mileage</th>
      <th>Driving_Experience_Years</th>
      <th>Vehicle_Type</th>
      <th>Vehicle_Age</th>
      <th>Credit_Score_Band</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POL10000</td>
      <td>CUST20000</td>
      <td>56</td>
      <td>Female</td>
      <td>Retired</td>
      <td>Newcastle</td>
      <td>7552</td>
      <td>36</td>
      <td>Sedan</td>
      <td>10</td>
      <td>Fair</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POL10001</td>
      <td>CUST20001</td>
      <td>53</td>
      <td>Female</td>
      <td>Unemployed</td>
      <td>Bristol</td>
      <td>13275</td>
      <td>31</td>
      <td>Motorcycle</td>
      <td>11</td>
      <td>Poor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POL10002</td>
      <td>CUST20002</td>
      <td>19</td>
      <td>Female</td>
      <td>Unemployed</td>
      <td>London</td>
      <td>12967</td>
      <td>0</td>
      <td>Sedan</td>
      <td>9</td>
      <td>Excellent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POL10003</td>
      <td>CUST20003</td>
      <td>77</td>
      <td>Female</td>
      <td>Retired</td>
      <td>Birmingham</td>
      <td>4346</td>
      <td>56</td>
      <td>Hatchback</td>
      <td>2</td>
      <td>Fair</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POL10004</td>
      <td>CUST20004</td>
      <td>24</td>
      <td>Male</td>
      <td>Employed</td>
      <td>Manchester</td>
      <td>9598</td>
      <td>2</td>
      <td>Sedan</td>
      <td>14</td>
      <td>Good</td>
    </tr>
  </tbody>
</table>
</div>


    Policyholders Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 11 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   Policy_ID                 5000 non-null   object
     1   Customer_ID               5000 non-null   object
     2   Age_of_Driver             5000 non-null   int64 
     3   Gender                    5000 non-null   object
     4   Occupation                5000 non-null   object
     5   Region                    5000 non-null   object
     6   Annual_Mileage            5000 non-null   int64 
     7   Driving_Experience_Years  5000 non-null   int64 
     8   Vehicle_Type              5000 non-null   object
     9   Vehicle_Age               5000 non-null   int64 
     10  Credit_Score_Band         5000 non-null   object
    dtypes: int64(4), object(7)
    memory usage: 429.8+ KB
    


    None


    Policyholders Describe:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age_of_Driver</th>
      <th>Annual_Mileage</th>
      <th>Driving_Experience_Years</th>
      <th>Vehicle_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.0000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48.7524</td>
      <td>12125.190800</td>
      <td>28.817600</td>
      <td>9.9910</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17.9704</td>
      <td>4036.274571</td>
      <td>17.919335</td>
      <td>6.0998</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.0000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.0000</td>
      <td>9388.250000</td>
      <td>13.000000</td>
      <td>5.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>49.0000</td>
      <td>12082.000000</td>
      <td>29.000000</td>
      <td>10.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>65.0000</td>
      <td>14860.750000</td>
      <td>45.000000</td>
      <td>15.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79.0000</td>
      <td>27410.000000</td>
      <td>61.000000</td>
      <td>20.0000</td>
    </tr>
  </tbody>
</table>
</div>


    Third-Parties Table Head:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Claim_ID</th>
      <th>TP_ID</th>
      <th>ThirdParty_Role</th>
      <th>TP_Injury_Severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CLM30000</td>
      <td>TP40000</td>
      <td>Pedestrian</td>
      <td>Minor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLM30002</td>
      <td>TP40001</td>
      <td>Passenger</td>
      <td>Minor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CLM30007</td>
      <td>TP40002</td>
      <td>Pedestrian</td>
      <td>Minor</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CLM30012</td>
      <td>TP40003</td>
      <td>Pedestrian</td>
      <td>Minor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CLM30015</td>
      <td>TP40004</td>
      <td>Driver</td>
      <td>Minor</td>
    </tr>
  </tbody>
</table>
</div>


    Third-Parties Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2410 entries, 0 to 2409
    Data columns (total 4 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   Claim_ID            2410 non-null   object
     1   TP_ID               2410 non-null   object
     2   ThirdParty_Role     2410 non-null   object
     3   TP_Injury_Severity  2410 non-null   object
    dtypes: object(4)
    memory usage: 75.4+ KB
    


    None


    Third-Parties Describe:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Claim_ID</th>
      <th>TP_ID</th>
      <th>ThirdParty_Role</th>
      <th>TP_Injury_Severity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2410</td>
      <td>2410</td>
      <td>2410</td>
      <td>2410</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>1997</td>
      <td>2410</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>CLM37990</td>
      <td>TP40000</td>
      <td>Passenger</td>
      <td>Minor</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>1</td>
      <td>836</td>
      <td>2050</td>
    </tr>
  </tbody>
</table>
</div>


## Missing Values Analysis

We examine missing values per column to determine completeness and potential issues.



```python
def missing_values(df):
    return (
        df.isnull()
        .sum()
        .to_frame("Missing_Count")
        .assign(Missing_Percent=lambda x: x.Missing_Count / len(df) * 100)
        .sort_values("Missing_Percent", ascending=False)
    )

# Apply function
missing_claims = missing_values(claims)
missing_policyholders = missing_values(policyholders)
missing_third_parties = missing_values(third_parties)

display(missing_claims)
display(missing_policyholders)
display(missing_third_parties)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing_Count</th>
      <th>Missing_Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ultimate_Claim_Amount</th>
      <td>425</td>
      <td>5.3125</td>
    </tr>
    <tr>
      <th>Settlement_Date</th>
      <td>425</td>
      <td>5.3125</td>
    </tr>
    <tr>
      <th>Claim_ID</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Policy_ID</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Accident_Date</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>FNOL_Date</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Claim_Type</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Claim_Complexity</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Fraud_Flag</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Litigation_Flag</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Estimated_Claim_Amount</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Severity_Band</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>Status</th>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing_Count</th>
      <th>Missing_Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Policy_ID</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Customer_ID</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Age_of_Driver</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Occupation</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Region</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Annual_Mileage</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Driving_Experience_Years</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Vehicle_Type</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Vehicle_Age</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Credit_Score_Band</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing_Count</th>
      <th>Missing_Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Claim_ID</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TP_ID</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ThirdParty_Role</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TP_Injury_Severity</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


## Target Variable Distribution

- The `Ultimate_Claim_Amount` is highly right-skewed.
- Most claims are low-cost, but a few very high-cost claims exist.
- These outliers are valid and retained.



```python
plt.figure(figsize=(12, 5))
plt.hist(claims["Ultimate_Claim_Amount"].dropna()/1000, bins=50, log=True, color='skyblue', edgecolor='black')
plt.xlabel("Ultimate Claim Amount (£k)")
plt.ylabel("Log Frequency")
plt.title("Ultimate Claim Amount Distribution (Log Scale) – To Detect Outliers")
plt.show()

```


    
![png](output_7_0.png)
    


## Schema Mapping

- `Claim_ID` links Claims → Third-Parties (one-to-many)
- `Policy_ID` links Claims → Policyholders (one-to-one)
- `Accident_Date` links Claims → Weather data (optional join)


## Key Data Issues

1. Missing `Ultimate_Claim_Amount` (~5.31%) – removed from training dataset.
2. Missing `Settlement_Date` (~5.31%) – retained as NaT to represent open claims.
3. Highly skewed target variable – requires robust modeling.
4. FNOL-stage data uncertainty – incomplete initial reports.


## 02_data_assessment
#### 1. Missing Values Assessment

##### An analysis of missing values was conducted across all datasets.

Missing Values Summary
Dataset	Column	Missing Count	Missing %	Treatment
Claims	Ultimate_Claim_Amount	425	5.31%	Rows excluded from model training
Claims	Settlement_Date	425	5.31%	Retained as NaT (open claims)
Policyholders	–	0	0%	No action required
Third-Parties	–	0	0%	No action required

Key Observations:
No missing values were found in the Policyholders or Third-Parties datasets.
Missing values in the Claims dataset are limited to settlement-related fields.
The missingness is expected and reflects claims that are still open.


#### 2. Target Variable Distribution & Skewness

##### The distribution of the Ultimate_Claim_Amount was examined using histograms with log-scaled frequency.

Observations:
The target variable exhibits a strong right-skewed distribution.
Most claims are low-cost and occur frequently.
A small number of high-cost claims create a long right tail.
Implications:
Extreme values (outliers) are expected in insurance claim data.
Models should be robust to skewness and outliers.
Consideration may be given to transformations (e.g., log transformation) during modeling.

#### 3. Presence of Extreme Outliers

Visual inspection of the target variable confirmed the presence of extreme high-value claims.
Outliers represent rare but financially significant events.
These claims are critical for reserve estimation and solvency planning.
Outliers were not removed, as they reflect real-world insurance risk.

#### 4. Key Data Issues Identified
Issue	Description	Impact
Incomplete settlement dates	Missing Settlement_Date for open claims	Expected; handled using NaT
Missing target values	Ultimate_Claim_Amount missing for unsettled claims	Rows excluded from training
Highly skewed target	Long right tail in claim amounts	Requires robust modeling
FNOL data uncertainty	Limited information available at FNOL stage	Motivates predictive modeling
    
#### 5. Summary
The dataset is largely complete, with missing values confined to settlement-related fields in the Claims dataset. The target variable is highly right-skewed with extreme outliers, which is typical in insurance data. Missing values were handled using domain-appropriate strategies to ensure modeling integrity, regulatory compliance, and realistic FNOL-stage prediction.


```python

```
