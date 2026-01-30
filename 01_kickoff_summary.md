#### Identify Tables

##### Claims Table – core claim details:
Claim_ID
Policy_ID
Accident_Date
FNOL_Date
Claim_Type (collision, theft, fire)
Estimated_Claim_Amount
Ultimate_Claim_Amount (final payout)
Settlement_Date

##### Policyholder Table – customer and vehicle info:
Policy_ID & Customer_ID
Driver_age
Driving_experience
Vehicle_type and Vehicle_age

##### Third-Party Involvement Table – accident severity:
Claim_ID (links to Claims)
Number_of_third_parties
Injury_severity (minor, serious, fatal)

#### Definitions

##### FNOL (First Notification of Loss)
The first report a customer makes to the insurer about an accident or loss.
It contains partial information about the incident and is used to start the claims process.
##### Estimated Claim Amount
The initial cost estimate of a claim at the FNOL stage.
Based on preliminary information, before full assessment.
##### Ultimate Claim Amount
The final payout for the claim after investigation, repairs, legal costs, and third-party involvement are accounted for.
##### Third-Party Severity
Indicates the seriousness of injuries or damages to other parties involved in the accident:
Minor → small or no medical attention
Serious → significant injuries or hospitalization
Fatal → death resulting from the accident

## Sample claim cost histogram


```python

```

## 01_kickoff_summary

#### Goal: Build a machine learning model to predict ultimate claim costs at FNOL stage.
#### Target: Accurate prediction of claim costs to improve reserve allocation and claims processing efficiency.
#### Why it matters:
###### Financial stability: prevent over/under-reserving
###### Operational efficiency: prioritize high-cost claims
###### Customer satisfaction: faster settlements and transparency

#### Metrics:
###### Reserve allocation accuracy improvement (%)
###### Prediction error metrics: MAE, RMSE, R²
###### Operational metrics: average time to settle claims

# 01_kickoff_summary
### 1. Goal:
##### To build a machine learning model that predicts the ultimate claim cost at the First Notification of Loss (FNOL) stage using historical claim, policy, and third-party data. 
##### The model will provide real-time predictions to claims handlers, allowing for more accurate reserve allocation, faster settlements, and improved financial stability.

### 2. Target:
##### Predict Ultimate Claim Amount for each claim.
##### Identify the most influential FNOL and policy variables affecting claim cost.
##### Improve reserve allocation accuracy by at least 15% compared to current benchmarks.

### 3. Why it Matters:
##### Financial stability: accurate cost predictions ensure adequate but efficient reserves.
##### Operational performance: claims teams can prioritize high-cost or complex claims earlier.
##### Customer satisfaction: early and transparent settlements reduce disputes and waiting times.
##### Competitive advantage: predictive analytics is increasingly standard in leading insurers.

### 4. Metrics:
##### Prediction accuracy metrics:Mean Absolute Error (MAE),Root Mean Squared Error (RMSE),R² Score
##### Reserve allocation improvement: percentage improvement compared to current estimates
##### Operational metrics:Average time to settle claims,Number of high-cost claims flagged early

### 5. Data Tables:
##### Claims:	(Core claim details)	Claim_ID, Policy_ID, Accident_Date, FNOL_Date, Claim_Type, Estimated_Claim_Amount, Ultimate_Claim_Amount, Settlement_Date
##### Policyholder:	(Customer & vehicle info)	Policy_ID, Customer_ID, Driver_Age, Driving_Experience, Vehicle_Type, Vehicle_Age
##### Third-Party Involvement:	(Accident severity)	Claim_ID, Number_of_Third_Parties, Injury_Severity


### 7. Key Definitions:
##### FNOL (First Notification of Loss): Initial report of an accident/loss.
##### Estimated Claim Amount: Preliminary claim cost estimate at FNOL.
##### Ultimate Claim Amount: Final payout after full investigation.
##### Third-Party Severity: Degree of injury or damage to other parties (Minor, Serious, Fatal).


```python

```


```python

```
