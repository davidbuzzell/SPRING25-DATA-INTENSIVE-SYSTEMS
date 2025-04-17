---
title: MBAI 417
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
  ## Please check in by creating an account and entering the provided code.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- H.02 is due tomorrow night @ 11:59 PM.
    - ~97% of you have already submitted! 🎉

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **EDA** concepts such as: 

  1. Handling incompatible data
  2. Database imputation
  3. Anomaly detection

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "header-slide">

# Why Data Cleaning?

</div>

<!--s-->

## Why Data Cleaning?

Data is often dirty! Don't ever give your machine learning or statistical model dirty data.

Remember the age-old adage:

> Garbage in, garbage out.

 Data cleaning is the process of converting source data into target data without errors, duplicates, or inconsistencies. You will often need to structure data in a way that is useful for analysis, so learning some basic data manipulation is **essential**.

<!--s-->

## Data Cleaning | Common Data Issues

1. ### Incompatible data
2. ### Missing values
3. ### Outliers

<!--s-->

<div class="header-slide">

# Incompatible Data

</div>

<!--s-->

## Handling Incompatible Data

<div style = "font-size: 0.85em;">

| Data Issue | Description | Example | Solution |
| --- | --- | --- | --- |
| Time / Date Unification | Date data is in different formats. | 10/11/2019 vs 11/10/2019 | Use libraries like <span class='code-span'>pandas</span> or <span class='code-span'>dateutil</span> for parsing and converting to ISO 8601 or UTC. |
| Character Representations | Text has different character encodings. | ASCII, UTF-8, ... | Use libraries like <span class='code-span'>chardet</span> or <span class='code-span'>unicodedata</span> to detect and convert encodings. |
| Unit Conversions | Numerical data conversions can be tricky. | 1 mile != 1.6 km | Use libraries like <span class='code-span'>pint</span> or <span class='code-span'>UnitConverter</span> to ensure accurate conversions. |
| Precision Representations | Data has variable precision. | 64-bit float to 16-bit integer | Use libraries like <span class='code-span'>numpy</span> to handle precision and ensure consistency. |
| Text Unification | Text data is in different formats. | D'Arcy; Darcy; DArcy; D Arcy; D\&\#39;Arcy | Use libraries like <span class='code-span'>re</span> for regex-based cleaning or <span class='code-span'>fuzzywuzzy</span> for text matching. |

</div>

<!--s-->

## Incompatible Data in OLAP Systems

Fortunately, OLAP systems often require data to be in a structured and consistent format (i.e. columns are usually type-specific). This means that you can often avoid many of the common data issues that arise in other types of data analysis. 

However, you may still encounter **any** of the issues described in the previous tables. For example, just requiring a float value in your column does not mean that the data will be the correct precision. Datetime columns don't usually enforce timezones.

<!--s-->

<div class="header-slide">

# Missing Values

</div>

<!--s-->

## Handling Missing Values

Missing values are a common issue in data analysis. They can occur for a variety of reasons, such as data entry errors, sensor failures, or simply because the data was not collected.

How do we handle missing values?

  ``` [1-5|4]
  id    col1    col2    col3    col4
  p1    2da6    0.99    32     43
  p2    cd2d    1.23    55      38
  p3    e53f    89.2    NaN     32
  p4    4e7c    0.72    9.7     35
  ```

<!--s-->

## Handling Missing Values

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

  Data is often missing from datasets. It's important to identify why it's missing. Once you have established that it is **missing at random**, you can proceed with **imputation**.

  When missing data, we have a few options at our disposal:

  1. Drop the entire row
  2. Drop the entire column
  3. Impute with a reasonable value

</div>
<div class="c2" style = "width: 50%">

  ``` 
  id    col1    col2    col3    col4
  p1    2da6    0.99    32     43
  p2    cd2d    1.23    55      38
  p3    e53f    89.2    NaN     32
  p4    4e7c    0.72    9.7     35
  ```
</div>
</div>

<!--s-->

## Missing at Random Assumption

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; font-size: 0.8em;">

### **Missing Completely at Random (MCAR)**: 
The missingness is completely random and unrelated to the data. This is the ideal scenario, but it is rare in practice. Little's MCAR test can be used to determine if data is MCAR.
### **Missing at Random (MAR)**
The missingness is related to other observed data, but not to the missing data itself. This is a common scenario and can be handled with imputation methods.
### **Missing Not at Random (MNAR)**
The missingness is related to the missing data itself. This is the worst-case scenario and can lead to biased imputation results.

</div>
<div class="c2" style = "width: 50%">

  <img src="https://pbs.twimg.com/media/D0kVZPPX0AA6xgd.png" width = "100%">
  <p style="text-align: center; font-size: 0.6em; color: grey;">McElreath 2019</p>

</div>
</div>

<!--s-->

## Handling Missing Values with Substitution


| Method | Description | When to Use |
| --- | --- | --- |
| Forward / backward fill | Fill missing value using the last / next valid value. | Time Series |
| Imputation by interpolation | Use interpolation to estimate missing values. | Time Series |
| Mean value imputation | Fill missing value with mean from column. | Random missing values |
| Conditional mean imputation | Estimate mean from other variables in the dataset. | Random missing values |
| Random imputation | Sample random values from a column. | Random missing values | 
| KNN imputation | Use K-nearest neighbors to fill missing values. | Random missing values |
| Random Forest Imputation | Uses random forest to fill missing values. | Random missing values |
| Multiple Imputation | Uses many regression models and other variables to fill missing values. | Random missing values |

<!--s-->

## L.05 | Q.01

You have streaming data that is occasionally dropping values. Which of the following methods would be appropriate to fill missing values when signal fails to update? 

*Please note, in this scenario, you can't use the future to predict the past.*

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%;">

<div style = "line-height: 2em;">
&emsp;A. Forward fill <br>
&emsp;B. Imputation by interpolation <br>
&emsp;C. Multiple imputation <br>
&emsp;D. Backward fill <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.05 | Q.01" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## L.05 | Q.02

You have a dataset with missing at random values (MAR). The dataset is not time series data. Which of the following methods would be appropriate to fill missing values?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%;">

<div style = "line-height: 2em;">
&emsp;A. Forward fill <br>
&emsp;B. Imputation by interpolation <br>
&emsp;C. Multiple imputation <br>
&emsp;D. Backward fill <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.05 | Q.02" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Local Imputation

Here are some examples of filling in missing values using Python's Pandas library and Scikit-learn. 

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

# Iterative imputation w/ random forest.
rf = RandomForestRegressor()
imputer = IterativeImputer(estimator=rf, max_iter=10, random_state=0)
df_mi = imputer.fit_transform(df)

# Multiple imputation w/ random forest.
df_mis = []
for random_state in range(10):
    rf = RandomForestRegressor(random_state=random_state)
    imputer = IterativeImputer(estimator=rf, max_iter=10, random_state=random_state)
    df_mi = imputer.fit_transform(df)
    df_mis.append(df_mi)
```

<!--s-->

## OLAP Imputation | Snowflake

Typically, multiple imputation is the best method for imputation. SnowFlake doesn't allow for true multiple imputation OOTB, but it does allow for iterative imputation w/ random state. Let's say you have a table in Snowflake with 1 million rows and 100 columns. You can use the <span class="code-span">snowflake.ml.modeling.impute.IterativeImputer</span> method in Snowflake to impute missing values, all while taking advantage of Snowflake's distributed architecture.

```sql
SELECT *
FROM snowflake.ml.modeling.impute.IterativeImputer(
  TABLE_NAME => 'my_table',
  MAX_ITER => 10,
  RANDOM_STATE => 0
);
```

By default, this method uses Bayesian ridge regression to impute missing values. You can also use other models, such as Random Forest or Gradient Boosting.

<!--s-->

## OLAP Imputation | BigQuery

BigQuery has a similar method for imputing missing values. You can use the <span class="code-span">ML.IMPUTER</span> function to impute missing values in your data. BigQuery (as of 03.2025) does not have support for more advanced imputation methods.

```sql
SELECT f, ML.IMPUTER(f, 'mean') OVER () AS output
FROM
  UNNEST([NULL, -3, -3, -3, 1, 2, 3, 4, 5]) AS f
ORDER BY f;
```

<!--s-->

<div class="header-slide">

# Anomaly Detection

</div>

<!--s-->

## Anomaly Detection

Outliers are extreme values that deviate from other observations on data. They can be caused by measurement error, data entry error, or they can be legitimate values. Outlier detection is, at its core, an anomaly detection problem. Here are some common methods for anomaly detection:

1. Statistical methods (e.g. Z-score, IQR)
2. Machine learning methods (e.g. Isolation Forest, One-Class SVM)
3. Visualization methods (e.g. box plots, scatter plots)

<!--s-->

## Identifying Anomalies with Z-Score

A z-score is a measure of how many standard deviations a data point is from the mean. A z-score of 0 indicates that the data point is exactly at the mean, while a z-score of 1.0 indicates that the data point is one standard deviation above the mean.

The formula for calculating the z-score is:

$$ z = \frac{x - \mu}{\sigma} $$

Where:
- $x$ is the data point
- $\mu$ is the mean of the data
- $\sigma$ is the standard deviation of the data

A z-score of 3 or -3 is often used as a threshold for identifying outliers. Please note, z-score is best used when the data is normally distributed.

<!--s-->

## Identifying Anomalies with IQR

The interquartile range (IQR) is a measure of statistical dispersion, or how spread out the data is. It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data.

The formula for calculating the IQR is:

$$ IQR = Q3 - Q1 $$

Where:
- $Q1$ is the 25th percentile of the data
- $Q3$ is the 75th percentile of the data

The IQR is often used to identify outliers. A common rule of thumb is that any data point that is more than 1.5 times the IQR above Q3 or below Q1 is considered an outlier. IQR is preferred over z-score when the data is not normally distributed.

<!--s-->

## Identifying Anomalies with Isolation Forest

An Isolation Forest is an unsupervised machine learning algorithm that is used for anomaly detection. It works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. 

This process is repeated recursively until all data points are isolated. The number of splits required to isolate a data point is called the path length. Anomalies are points that have shorter path lengths, as they are easier to isolate.

<div style="text-align: center;">
<img src = "https://spotintelligence.com/wp-content/uploads/2024/05/illustration-isolation-forest.jpg" width = "70%" style="border-radius: 10px; margin: 0;">
<p style="text-align: center; font-size: 0.6em; color: grey; margin: 0;">Spot Intelligence 2024</p>
</div>

<!--s-->

## Identifying Anomalies with Visualization

Often, the best way to identify anomalies is through visualization. This is commonly done with 1D data through boxplots.

<img src = "https://miro.medium.com/v2/resize:fit:1400/1*0MPDTLn8KoLApoFvI0P2vQ.png" width = "100%" style="border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Agarwal 2019</p>

<!--s-->

## L.05 | Q.03

You are working for a cybersecurity company tasked with detecting anomalies in network traffic data. The dataset contains hundreds of features, such as packet size, source and destination IPs, protocols, and timestamps. The data is high-dimensional and not normally distributed.

Select the most appropriate method for detecting anomalies in this dataset.

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%;">

<div style = "line-height: 2em;">
&emsp;A. IQR <br>
&emsp;B. Z-Score <br>
&emsp;C. Isolation Forest <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.05 | Q.03" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Anomaly Detection | OLAP Example

You can use [Snowflake ML Functions](https://docs.snowflake.com/user-guide/ml-functions/anomaly-detection) to detect anomalies in your tables. By default, this method will use a [gradient boosting machine](https://docs.snowflake.com/user-guide/ml-functions/anomaly-detection).

1. Create a model using the <span class="code-span">SNOWFLAKE.ML.ANOMALY_DETECTION</span> function.

```sql
CREATE OR REPLACE SNOWFLAKE.ML.ANOMALY_DETECTION anomaly_detector(
  INPUT_DATA =>
    TABLE(SELECT date, sales FROM historical_sales_data),
  TIMESTAMP_COLNAME => 'date',
  TARGET_COLNAME => 'sales',
  LABEL_COLNAME => '');
```

2. Create a view with the data you want to analyze.
```sql
CREATE OR REPLACE VIEW view_with_data_to_analyze
  AS SELECT date, sales FROM new_sales_data;
```

3. Call the model to detect anomalies in the data.
```sql
CALL anomaly_detector!DETECT_ANOMALIES(
  INPUT_DATA => TABLE(view_with_data_to_analyze),
  TIMESTAMP_COLNAME =>'date',
  TARGET_COLNAME => 'sales'
);
```

<!--s-->

## Summary

- Incompatible data can often be handled with unit conversions, precision representations, character representations, text unification, and time/date unification.

- Missing values can be handled by a broad variety of methods, but multiple imputation is often the best method. OLAP imputation can be done with the <span class='code-span'>snowflake.ml.modeling.impute.IterativeImputer method</span>.

- Anomalies can be detected with a variety of methods. OLAP anomaly detection can be done with the <span class='code-span'>SNOWFLAKE.ML.ANOMALY_DETECTION</span> method.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **EDA** concepts such as: 

  1. Handling incompatible data
  2. Database imputation
  3. Anomaly detection

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->