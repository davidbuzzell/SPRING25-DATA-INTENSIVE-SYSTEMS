---
title: MBAI
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Data Intensive Systems
  ## L.05 | OLAP + EDA I

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 10%">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Welcome to Data Intensive Systems.
  ## Please check in by creating an account and entering the code on the chalkboard.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- 

<!--s-->

<div class="header-slide">

# OLAP + EDA I

</div>

<!--s-->

## Agenda

### Handling Incompatible Data

**Scenario**: You work for a large tech company. Your team is tasked with analyzing user data from a web application, but the data is in different formats.

- What is incompatible data?
- How do we handle incompatible data?

### Database Imputation

**Scenario**: You work for a green energy startup that is optimizing green energy production. You are tasked with analyzing sensor data from solar panels, but some of the data is missing.

- What is imputation?
- Why do we care about imputation?
- How do we impute missing data?

### Anomaly Detection

**Scenario**: You work at a small e-commerce company. Your team is tasked with analyzing customer purchase data to identify any unusual spending patterns that may indicate fraudulent activity.

- What is an outlier?
- Why do we care about outliers?
- How do we automatically detect outliers?

<!--s-->

<div class = "header-slide">

# Data Cleaning

</div>

<!--s-->

## Data Cleaning

Data is often dirty! Don't ever give your machine learning or statistical model dirty data.

Remember the age-old adage:

> Garbage in, garbage out.

 Data cleaning is the process of converting source data into target data without errors, duplicates, or inconsistencies. You will often need to structure data in a way that is useful for your analysis, so learning some basic data manipulation is **essential**.

<!--s-->

## Data Cleaning | Common Data Issues

1. Incompatible data
2. Missing values
3. Extreme Outliers

<!--s-->

## Data Cleaning | Handling Incompatible Data

<div style = "font-size: 0.85em;">

| Data Issue | Description | Example | Solution |
| --- | --- | --- | --- |
| Unit Conversions | Numerical data conversions can be tricky. | 1 mile != 1.6 km | Measure in a common unit, or convert with caution. |
| Precision Representations | Data can be represented differently in different programs. | 64-bit float to 16-bit integer | Use the precision necessary and hold consistent. |
| Character Representations | Data is in different character encodings. | ASCII, UTF-8, ... | Create using the same encoding, or convert with caution. |
| Text Unification | Data is in different formats. | D'Arcy; Darcy; DArcy; D Arcy; D&-#-3-9-;Arcy | Use a common format, or convert with caution. <span class="code-span">RegEx</span> will be your best friend.| 
| Time / Date Unification | Data is in different formats. | 10/11/2019 vs 11/10/2019 | Use standard libraries & UTC. A personal favorite is seconds since epoch. |

</div>

<!--s-->

## Data Cleaning | Handling Missing Values

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

  Data is often missing from datasets. It's important to identify why it's missing. Once you have established that it is **missing at random**, you can proceed with **substitution**.

  When missing data, we have a few options at our disposal:

  1. Drop the entire row
  2. Drop the entire column
  3. Substitute with a reasonable value

</div>
<div class="c2" style = "width: 50%">

  ``` [1-5|4]
  id    col1    col2    col3    col4
  p1    2da6    0.99    32     43
  p2    cd2d    1.23    55      38
  p3    9999    89.2    NaN     32
  p4    4e7c    0.72    9.7     35
  ```
</div>
</div>

<!--s-->

## Data Cleaning | Missing at Random Assumption

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

  - **Missing Completely at Random (MCAR)**: The missingness is completely random and unrelated to the data. This is the ideal scenario, but it is rare in practice. Little's MCAR test can be used to determine if data is MCAR.
  - **Missing at Random (MAR)**: The missingness is related to other observed data, but not to the missing data itself. This is a common scenario and can be handled with imputation methods.
  - **Missing Not at Random (MNAR)**: The missingness is related to the missing data itself. This is the worst-case scenario and can lead to biased results.

</div>
<div class="c2" style = "width: 50%">

  <img src="https://pbs.twimg.com/media/D0kVZPPX0AA6xgd.png" width = "100%">
  <p style="text-align: center; font-size: 0.6em; color: grey;">McElreath 2019</p>

</div>
</div>

<!--s-->

## Data Cleaning | Handling Missing Values with Substitution

<div class = "col-wrapper"> 
<div class = "c1" style = "width: 70%; font-size: 0.85em;">


| Method | Description | When to Use |
| --- | --- | --- |
| Forward / backward fill | Fill missing value using the last / next valid value. | Time Series |
| Imputation by interpolation | Use interpolation to estimate missing values. | Time Series |
| Mean value imputation | Fill missing value with mean from column. | Random missing values |
| Conditional mean imputation | Estimate mean from other variables in the dataset. | Random missing values |
| Random imputation | Sample random values from a column. | Random missing values | 
| KNN imputation | Use K-nearest neighbors to fill missing values. | Random missing values |
| Multiple Imputation | Uses many regression models and other variables to fill missing values. | Random missing values |

</div>

<div class = "c2 col-centered" style = "width: 40%">

```python
import pandas as pd
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Load data.
df = pd.read_csv('data.csv')

# Forward fill.
df_ffill = df.fillna(method='ffill')

# Backward fill.
df_bfill = df.fillna(method='bfill')

# Mean value imputation.
df_mean = df.fillna(df.mean())

# Random value imputation.
df_random = df.fillna(df.sample())

# Imputation by interpolation.
df_interpolate = df.interpolate()

# KNN imputation.
imputer = KNNImputer(n_neighbors=5)
df_knn = imputer.fit_transform(df)

# Multiple imputation w/ random forest.
rf = RandomForestRegressor()
imputer = IterativeImputer(estimator=rf, max_iter=10, random_state=0)
df_mi = imputer.fit_transform(df)

```

</div>
</div>

<!--s-->

## Data Cleaning | Q.01

You have streaming data that is occasionally dropping values. Which of the following methods would be appropriate to fill missing values when signal fails to update? 

*Please note, in this scenario, you can't use the future to predict the past.*

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 60%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Forward fill <br>
&emsp;B. Imputation by interpolation <br>
&emsp;C. Mean value imputation <br>
&emsp;D. Backward fill <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 40%; padding-bottom: 20%">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Q.01" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## OLAP Imputation

Typically, multiple imputation is the best method for imputation. Let's say you have a table in Snowflake with 1 million rows and 100 columns. You can use the snowflake.ml.modeling.impute.IterativeImputer method in Snowflake to impute missing values, all while taking advantage of Snowflake's distributed architecture.

```sql
SELECT *
FROM snowflake.ml.modeling.impute.IterativeImputer(
  TABLE_NAME => 'my_table',
  MAX_ITER => 10,
  RANDOM_STATE => 0
);
```

By default, this method uses Bayesian Ridge regression to impute missing values. You can also use other regression models, such as Random Forest or Gradient Boosting.

<!--s-->

## Data Cleaning | Anomaly Detection

Outliers are extreme values that deviate from other observations on data. They can be caused by measurement error, data entry error, or they can be legitimate values. Outlier detection is, at its core, an anomaly detection problem. Here are some common methods for anomaly detection:

1. Statistical methods (e.g. Z-score, IQR)
2. Machine learning methods (e.g. Isolation Forest, One-Class SVM)
3. Visualization methods (e.g. box plots, scatter plots)

<!--s-->

## Data Cleaning | Identifying Anomalies with Z-Score

A z-score is a measure of how many standard deviations a data point is from the mean. A z-score of 0 indicates that the data point is exactly at the mean, while a z-score of 1.0 indicates that the data point is one standard deviation above the mean.

The formula for calculating the z-score is:

$$ z = \frac{x - \mu}{\sigma} $$

Where:
- $x$ is the data point
- $\mu$ is the mean of the data
- $\sigma$ is the standard deviation of the data

A z-score of 3 or -3 is often used as a threshold for identifying outliers.

<!--s-->

## Data Cleaning | Identifying Anomalies with IQR

The interquartile range (IQR) is a measure of statistical dispersion, or how spread out the data is. It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data.

The formula for calculating the IQR is:

$$ IQR = Q3 - Q1 $$

Where:
- $Q1$ is the 25th percentile of the data
- $Q3$ is the 75th percentile of the data

The IQR is often used to identify outliers. A common rule of thumb is that any data point that is more than 1.5 times the IQR above Q3 or below Q1 is considered an outlier. IQR is preferred over z-score when the data is not normally distributed.

<!--s-->

## Data Cleaning | Identifying Anomalies with Isolation Forest

An Isolation Forest is an unsupervised machine learning algorithm that is used for anomaly detection. It works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. This process is repeated recursively until all data points are isolated. The number of splits required to isolate a data point is called the path length. Anomalies are points that have shorter path lengths, as they are easier to isolate.

<img src = "https://spotintelligence.com/wp-content/uploads/2024/05/illustration-isolation-forest.jpg" width = "100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Spot Intelligence 2024</p>

<!--s-->

## Data Cleaning | Identifying Anomalies with One-Class SVM

A One-Class SVM is a supervised machine learning algorithm that is used for anomaly detection. It works by finding a hyperplane that separates the data points from the origin. The distance from the hyperplane to the origin is called the decision function. Anomalies are points that have a decision function value less than a certain threshold.

<img src = "https://pub.mdpi-res.com/applsci/applsci-13-01734/article_deploy/html/images/applsci-13-01734-g001.png?1675319625" width = "100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">MDPI 2023</p>

<!--s-->

## Data Cleaning | Identifying Anomalies with Visualization

Often, the best way to identify anomalies is through visualization. This is commonly done through boxplots.

<img src = "https://miro.medium.com/v2/resize:fit:1400/1*0MPDTLn8KoLApoFvI0P2vQ.png" width = "100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Agarwal 2019</p>

<!--s-->

## OLAP Anomaly Detection

You can use the snowflake.ml.modeling.anomaly_detection.IsolationForest method in Snowflake to detect anomalies in your data. This method uses the Isolation Forest algorithm to identify anomalies.

```sql

SELECT *
FROM snowflake.ml.modeling.anomaly_detection.IsolationForest(
  TABLE_NAME => 'my_table',
  MAX_ITER => 100,
  RANDOM_STATE => 0
);
```
By default, this method uses the Isolation Forest algorithm to identify anomalies. You can also use other algorithms, such as One-Class SVM.

<!--s-->

## Summary

- Incompatible data can often be handled with unit conversions, precision representations, character representations, text unification, and time/date unification.

- Missing values can be handled by a broad variety of methods, but multiple imputation is often the best method. OLAP imputation can be done with the snowflake.ml.modeling.impute.IterativeImputer method.

- Anomalies can be detected with a variety of methods. OLAP anomaly detection can be done with the snowflake.ml.modeling.anomaly_detection.IsolationForest method.

<!--s-->

