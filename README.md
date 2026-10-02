# Early Prediction of Motor Insurance Claim Cost Using FNOL Data

This repository contains the notebooks,markdown and pdf documentation for a machine learning project that predicts ultimate motor insurance claim cost at the FNOL stage.
The project combines claims, policyholder, and third-party data to enable early claim severity prediction, with strong focus on feature engineering. data leakage prevention, and model evaluation.

----

# Early Prediction of Motor Insurance Claim Cost Using FNOL Data

## Project Overview

This project uses machine learning to predict the ultimate cost of motor insurance claims at the First Notice of Loss (FNOL) stage.

The project combines claims, policyholder and third-party data to estimate claim severity using information available early in the claims lifecycle. A key focus was **feature engineering and data leakage prevention** to ensure that the model only uses information that would realistically be available when a claim is first reported.

The analysis aims to demonstrate how predictive analytics can support earlier claims assessment, prioritisation and reserve planning.

## Business Problem

Insurers often have limited information when a claim is first reported, while the eventual claim cost can vary significantly depending on factors such as claim type, vehicle characteristics, accident circumstances and third-party involvement.

Accurate early estimation of claim severity can help insurers:

* Identify potentially high-cost claims earlier
* Support claims triage and prioritisation
* Improve the allocation of reserves
* Support more consistent claims assessment
* Provide earlier insight into potential claims exposure

The challenge is to make these predictions using **FNOL-stage information without introducing data leakage from events that occur later in the claims lifecycle**.

## Project Objectives

* Develop a machine learning model to predict ultimate motor insurance claim cost using FNOL-stage data.
* Identify the claims, policyholder and third-party variables most associated with claim severity.
* Compare different regression models using appropriate performance metrics.
* Apply feature engineering and data leakage controls to improve the reliability of the modelling process.
* Evaluate how the resulting model could support early claims assessment and prioritisation.

## Data

The project uses three datasets:

### Claims Data

Contains claim and early-stage claims information, including:

* Claim ID
* Policy ID
* Accident date
* FNOL date
* Claim type
* Claim complexity
* Estimated claim amount
* Ultimate claim amount
* Severity band
* Status

### Policyholder Data

Contains customer and policy characteristics, including:

* Policy ID
* Customer ID
* Driver age
* Gender
* Occupation
* Region
* Annual mileage
* Driving experience
* Vehicle type
* Vehicle age
* Credit score band

### Third-Party Data

Contains information relating to third-party involvement and accident severity, including:

* Claim ID
* Third-party ID
* Third-party role
* Third-party injury severity

### Data Relationships

* Claims → Policyholder: one-to-one
* Claims → Third-party records: one-to-many

## Data Preparation & Feature Engineering

The datasets were integrated using claim and policy identifiers before being prepared for modelling.

Key activities included:

* Data quality assessment
* Missing-value treatment
* Outlier assessment
* Categorical variable encoding
* Feature engineering
* Feature selection
* Model-ready dataset preparation
* Validation of feature availability at FNOL

Particular attention was given to **data leakage prevention**. Variables representing information that would only become available later in the claims lifecycle were excluded from the predictive feature set.

## Modelling Approach

Several regression models were developed and compared:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

Models were evaluated using:

* **RMSE** – Root Mean Squared Error
* **MAE** – Mean Absolute Error
* **R²** – Coefficient of Determination

A time-aware train/test split based on accident date was used to provide a more realistic assessment of how the model could perform on future claims.

## Key Analysis & Results

The modelling process focused on comparing baseline and more advanced machine learning approaches to determine how effectively FNOL-stage information could explain variation in ultimate claim cost.

The project also examined the variables contributing most strongly to predicted claim severity and assessed model performance on unseen data.

Detailed model comparisons, feature analysis and visualisations are available in the project notebooks.

## Data Leakage Prevention

Data leakage was treated as a key modelling risk because the target variable represents the **ultimate claim cost**, which is only known after the claim develops and settles.

To reduce this risk:

* Post-settlement information was excluded.
* Future claim outcomes were excluded from the predictive features.
* Features were assessed for their availability at FNOL.
* Accident-date-based train/test splitting was used.
* The final feature set was restricted to information that could reasonably be available when the claim is first reported.

## Business Impact

The project demonstrates how machine learning could support insurance claims teams by providing an early estimate of potential claim severity.

Potential applications include:

* **Claims triage:** prioritising potentially high-cost claims for earlier review.
* **Reserve planning:** providing an additional early indication of expected claim exposure.
* **Claims management:** supporting more consistent early-stage assessment.
* **Risk identification:** highlighting claims with characteristics associated with higher expected costs.
* **Operational efficiency:** reducing reliance on purely manual early-stage assessment.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Jupyter Notebook

## Project Structure

```text
early-motor-claim-cost-prediction/

├── data/
│   ├── claims.xlsx
│   ├── policyholders.xlsx
│   └── third_parties.xlsx
│
├── notebooks/
│   ├── 01_kickoff_summary.ipynb
│   ├── 02_data_assessment.ipynb
│   ├── 03_cleaning.ipynb
│   ├── 04_features_eda.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_advanced_models.ipynb
│   └── 07_model_evaluation.ipynb
│
└── README.md
```



[Early Prediction of Motor Insurance Claim Cost Using FNOL Data.docx](https://github.com/user-attachments/files/26087105/Early.Prediction.of.Motor.Insurance.Claim.Cost.Using.FNOL.Data.docx)
