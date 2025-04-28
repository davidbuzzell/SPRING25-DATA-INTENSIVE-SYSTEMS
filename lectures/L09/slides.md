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
  ## L.09 | Distributed Data Processing

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

## Midterm Feedback Actions

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 5%;">

### Enhance Material Accessibility

I'll provide downloadable PDF versions of slides for easy reference and annotation. Click on the [printable] link in the README, and this will produce a PDF version of the slides that you can download / print.

### Increase / Continue Interactive and Applied Learning

I'll work to include more interactive elements, such as live coding sessions, during lectures.

</div>
<div class="c2" style = "width: 50%">

### Exam Preparation Concerns:

We'll have a review session before the midterm exam, and I'll highlight exactly what I want you to focus on (and provide sample questions during the review session).

</div>
</div>

<!--s-->

## Announcements

- Video recordings have been figured out! From this lecture forward, go to the Zoom tab on Canvas to find our recordings in Ford. Global Hub recordings will be available in the Panopto folder on Canvas.
  - If you have any questions after reviewing slides from the unrecorded lectures, please feel free to reach out to me.

- H.03 is released today and due on 05.05.2025 @ 11:59PM.


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with preprocessing raw data to create a machine learning-ready dataset?


  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Data Processing

</div>

<!--s-->

# Agenda

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

## Data Processing
- ### Creating an ML-Ready Dataset
- ### Adding Interaction Terms
- ### Transformations
- ### Feature Selection

</div>
<div class="c2" style = "width: 50%">

## Distributed Data Processing
- ### Creating a Dataset w/ Snowflake

</div>
</div>

<!--s-->

<div class="header-slide">

# Data Processing

</div>

<!--s-->

## Overview

Often, your goal is to build a complete, numerical matrix with rows (instances) and columns (features).

- **Ideal**: Raw Dataset → Model → Task
- **Reality**: Raw Dataset → ML-Ready Dataset → Features (?) → Model → Task

Once you have a numerical matrix, you can apply modeling methods to it.

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

<!--s-->

## Overview | From Raw Dataset to ML-Ready Dataset

The raw dataset is transformed into a machine learning-ready dataset by converting the data into a format that can be used by machine learning models. This typically involves converting the data into numerical format, removing missing values, and scaling the data.

Each row should represent an instance (e.g. a student) and each column should represent a consistently-typed and formatted feature (e.g. homework score, exam score). By convention, the final column often represents the target variable (e.g. final exam score).

<br><br><br>

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em; margin-right: 5%;">

### Raw Dataset:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| "2549@nw.edu" | 90%| 89% | None | 93% |
| "7856@nw.edu" | 79%| 75% | 70% | 65% |
| ... | ... | ... | ... | ... |

</div>
<div class="c2" style = "width: 50%;  font-size: 0.8em;">

### ML-Ready:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

</div>
</div>

<!--s-->

<div class="header-slide">

# Interaction Terms

</div>

<!--s-->

## Interaction Terms

Interaction terms are used to capture the combined effect of two or more features on the target variable. They are created by multiplying two or more features together.

- **Example**: If you have two features, <span class="code-span">x1</span> and <span class="code-span">x2</span>, the interaction term would be <span class="code-span">x1 * x2</span>.
- **Why**: Interaction terms can help to capture non-linear relationships between features and the target variable.
- **How**: You can create interaction terms using the <span class="code-span">PolynomialFeatures</span> class from the <span class="code-span">sklearn.preprocessing</span> module.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%;  font-size: 0.8em;margin-right: 5%;">

### ML-Ready:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.73 |
| ... | ... | ... | ... | ... |


</div>
<div class="c2" style = "width: 50%;  font-size: 0.8em;">

### ML-Ready + Interaction Term:

| Student ID | Homework | Exam 1 | Exam 2 | Average Exam Score | Final |
|------------|----------|--------|--------|---------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.74 | 0.65 |
| ... | ... | ... | ... | ... | ... |

</div>
</div>

<!--s-->

## Interaction Terms | PolynomialFeatures

PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form 

$$[a, b]$$

the degree-2 polynomial features are 

$$ [1, a, b, a^2, ab, b^2] $$

this is a very quick way to add interaction terms to your dataset.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(degree=2, interaction_only=False,  include_bias=True)
poly.fit_transform(X)
```

```
# Original
[[0 1]
 [2 3]
 [4 5]]
```

```
# Transformed
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
```

<!--s-->

<div class="header-slide">

# Transformations

</div>

<!--s-->

## Data Transformations

Here are a number of transformations and feature engineering methods that are common in the data preprocessing stage.

<div style="font-size: 0.75em">

| Type | Transformation | Description |
|-----------|----------------|-------------|
| Numerical | Binarization | Convert numeric to binary. |
| Numerical | Binning | Group numeric values. |
| Numerical | Log Transformation | Manage data scale disparities. |
| Numerical | Scaling | Standardize or scale features. |
| Categorical | One-Hot Encoding | Convert categories to binary columns. |
| Categorical | Feature Hashing | Compress categories into hash vectors. |
| Temporal | Temporal Binning & Standardization | Convert to bins, manage time zones. |
| Temporal | Lag Features | Create lagged variables. |
| Temporal | Rolling Features | Create rolling averages. |
| Temporal | Frequency Features | Create frequency-based features. |
| Text | Chunking | Split text into chunks. |
| Text | Embedding | Convert text to vectors. |
| Missing | Drop | Remove missing values. |
| Missing | Imputation | Fill missing values. |

</div>

<!--s-->

<div class="header-slide">

# Numerical Data

</div>

<!--s-->

## Numerical Data | Binarization

Convert numerical values to binary values via a threshold. Values above the threshold are set to 1, below threshold are set to 0.

```python
from sklearn.preprocessing import Binarizer

transformer = Binarizer(threshold=3).fit(data)
transformer.transform(data)
```

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binarized.png" width="800" style="display: block; margin: 0 auto; border-radius: 10px;">

<!--s-->

## Numerical Data | Binning

Group numerical values into bins.

- **Uniform**: Equal width bins. Use this when the data is uniformly distributed.
- **Quantile**: Equal frequency bins. Use this when the data is not uniformly distributed, or when you want to be robust to outliers.

```python
from sklearn.preprocessing import KBinsDiscretizer

# uniform binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit(data)
transformer.transform(data)

# quantile binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit(data)
transformer.transform(data)
```

<!--s-->

## Numerical Data | Binning

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binning.png" style="display: block; margin: 0 auto; border-radius: 10px;">

<!--s-->

## Numerical Data | Log Transformation

Logarithmic transformation of numerical values. This is useful for data with long-tailed distributions, or when the scale of the data varies significantly.


```python
import numpy as np
transformed = np.log(data)
```

<br><br>

<img src="https://storage.googleapis.com/slide_assets/long-tail.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Komorowski, 2016</p>


<!--s-->

## Numerical Data | Scaling

Standardize or scale numerical features.

- **MinMax**: Squeeze values into [0, 1].

$$ x_{\text{scaled}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} $$

- **Standard**: Standardize features to have zero mean and unit variance:

$$ x_{\text{scaled}} = \frac{x - \mu}{\sigma} $$

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# min-max scaling
scaler = MinMaxScaler().fit(data)
scaler.transform(data)

# standard scaling
scaler = StandardScaler().fit(data)
scaler.transform(data)
```

<!--s-->

## Numerical Data | Scaling

| Method | Pros | Cons | When to Use | 
|--------|------|------| ------------|
| MinMax | Bounded between 0 and 1. | Sensitive to outliers | When the data is uniformly distributed. |
| Standard | More robust to outliers. | Not bounded. | When the data is normally distributed. |

**A special note on training / testing sets** -- always fit the scaler on the training set and transform both the training and testing sets using those parameters. This ensures that the testing set is not used to influence the training set.

<!--s-->

## Numerical Data | Why Scale Data?

### Convergence
Some models (like SVMs and neural networks) converge faster when the data is scaled. Scaling the data can help the model find the optimal solution more quickly.

### Performance
Some models (like KNN) are sensitive to the scale of the data. Scaling the data can improve the performance of these models. Consider the KNN model -- if the data is not scaled, the model will give more weight to features with larger scales.

### Regularization
Regularization methods (like L1 and L2 regularization) penalize large coefficients. If the features are on different scales, the regularization term may penalize some features more than others.

<!--s-->

<div class="header-slide">

# Categorical Data

</div>

<!--s-->

## Categorical Data | One-Hot Encoding

Convert categorical features to binary columns. This will create a binary column for each unique value in the data. For example, the color feature will be transformed into three columns: red, green, and blue.

```python
from sklearn.preprocessing import OneHotEncoder

data = [["red"], ["green"], ["blue"]]
encoder = OneHotEncoder().fit(data)
encoder.transform(data)
```


```
array([[0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.]])
```

<!--s-->

## Categorical Data | One-Hot Encoding w/ Pandas

Pandas has a built-in function to perform one-hot encoding. The example below uses the get_dummies function to create binary columns for each unique value in the color feature.

```python

import pandas as pd
data = pd.DataFrame({"color": ["red", "green", "blue"], "value": [1, 2, 3]})
data = pd.get_dummies(data, columns=["color"], dtype = "int")
data.head()
```

<div class = "col-wrapper">
<div class="c1" style = "width: 30%; margin: 5%;">

### Original Data

| color | value |
|------------|-------|
| red        | 1     |
| green      | 2     |
| blue       | 3     |

</div>
<div class="c2" style = "width: 70%; margin: 5%">

### Transformed Data

| color_blue | color_green | color_red | value |
|------------|-------------|-----------| ------|
| 0          | 0           | 1         | 1 |
| 0          | 1           | 0         | 2 |
| 1          | 0           | 0         | 3 |

</div>
</div>

<!--s-->

## Categorical Data | Feature Hashing

Feature hashing reduces the dimensionality of the feature space. Hashing is using a function that maps categorical values to fixed-length vectors.

Compared to one-hot encoding, feature hashing is more memory-efficient and can handle high-dimensional data. It is typically used when some categories are unknown ahead of time, or when the number of categories is very large.

However, it can lead to collisions, where different categories are mapped to the same column. In this case below, we are hashing the color feature into a 2-dimensional vector. 


```python
from sklearn.feature_extraction import FeatureHasher

data = [{"color": "red"}, {"color": "green"}, {"color": "blue"}]
hasher = FeatureHasher(n_features=2, input_type='dict').fit(data)
hasher.transform(data)
```

```
array([[-1,  0],
       [ 0,  1],
       [ 1,  0]])
```

<!--s-->

## L.09 | Q.01

<div class='col-wrapper' style = 'display: flex; align-items: top;'>
<div class='c1' style = 'width: 60%; display: flex; flex-direction: column;'>

You are working with a dataset that contains a feature with 100 unique categories. You are unsure if all categories are present in the training set, and you want to reduce the dimensionality of the feature space. Which method would you use?

<div style = 'line-height: 2em;'>
&emsp;A. One-Hot Encoding <br>
&emsp;B. Feature Hashing <br>
</div>
</div>
<div class="c2" style="width: 50%; height: 100%;">
<iframe src="https://drc-cs-9a3f6.web.app/?label=L.09 | Q.01" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Temporal Data

</div>

<!--s-->

## Temporal Data | Temporal Granularity

Converting temporal features to bins based on the required granularity for your model helps manage your time series data. For example, you can convert a date feature to year / month / day, depending on the required resolution of your model.

We'll cover more of time series data handling in our time series analysis lecture -- but pandas has some great tools for this!

```python
import pandas as pd

data = pd.DataFrame({"date": ["2021-01-01", "2021-02-01", "2021-03-01"]})
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day

```

|date|month|day|
|---|---|---|
|2021-01-01|1|1|
|2021-02-01|2|1|
|2021-03-01|3|1|

<!--s-->

## Temporal Data | Lag Features

Lag features are used to create new features based on previous values of a time series. This is useful for capturing temporal dependencies in the data.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python

import pandas as pd

data = pd.DataFrame({"date": ["2021-01-01", "2021-02-01", "2021-03-01"], "value": [1, 2, 3]})
data["date"] = pd.to_datetime(data["date"])

data["lag_1"] = data["value"].shift(1)
data["lag_2"] = data["value"].shift(2)
```

</div>
<div class="c2" style = "width: 50%">

date|value|lag_1|lag_2
---|---|---|---
2021-01-01|1|NaN|NaN
2021-02-01|2|1.0|NaN
2021-03-01|3|2.0|1.0

</div>
</div>

<!--s-->

## Temporal Data | Rolling Features

Rolling features are used to create new features based on the rolling average of a time series. This is useful for smoothing out noise in the data.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

```python
import pandas as pd
data = pd.DataFrame({"date": ["2021-01-01", "2021-02-01", "2021-03-01"], "value": [1, 2, 3]})
data["date"] = pd.to_datetime(data["date"])
data["rolling_mean"] = data["value"].rolling(window=2).mean()
```
</div>
<div class="c2" style = "width: 50%">

date|value|rolling_mean
---|---|---
2021-01-01|1|NaN
2021-02-01|2|1.5
2021-03-01|3|2.5

</div>

</div>

<!--s-->

## Temporal Data | Frequency Features

Sometimes, we have time series data that is noisy. We can utilize filtering methods to create new features based on the frequency of the data. Here is an example of cleaning up a noisy sine wave with a low-pass filter.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)
b, a = signal.butter(3, 0.1) # 3rd order butterworth filter.
y_filtered = signal.filtfilt(b, a, y)
plt.plot(x, y, label='Noisy')
plt.plot(x, y_filtered, label='Filtered')
plt.legend()
```

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/noisy_sin_wave.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'></p>
</div>

</div>
</div>


<!--s-->

<div class="header-slide">

# Text Data

</div>

<!--s-->

## Text Data | Chunking

Chunking is the process of splitting text into smaller, more manageable pieces. This is useful for processing large amounts of text data. NLTK has a built-in chunking function that can be used to split text into sentences or words. We will talk more about chunking in the RAG lecture.

```python

import nltk
from nltk import sent_tokenize, word_tokenize

text = "This is a sentence. This is another sentence."
sentences = sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]
```

<!--s-->

## Text Data | Embedding

Embedding is the process of converting text into numerical vectors. This is useful for processing text data with machine learning models, because we can use the embeddings as features. We talked about embeddings in L.08.

```python
import openai
import numpy as np
import pandas as pd

data = pd.DataFrame({"text": ["This is a sentence.", "This is another sentence."]})
data["embedding"] = data["text"].apply(lambda x: np.array(openai.Embedding.create(input=x, model="text-embedding-ada-002")["data"][0]["embedding"]))
```

<!--s-->

<div class="header-slide">

# Missing Data

</div>

<!--s-->

## Missing Data | Drop

Dropping missing values is the simplest method for handling missing data. However, this can lead to loss of information and bias in the data.

```python

import pandas as pd

data = pd.DataFrame({"A": [1, 2, None], "B": [4, 5, 6]})
data.dropna()
```

<!--s-->

## Missing Data | Imputation

We covered imputation in more detail in L.03.

<div style = "font-size: 0.85em;">


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

<!--s-->

<div class="header-slide">

# Feature Selection

</div>

<!--s-->

## Feature Selection | Chi-Squared

One method to reduce dimensionality is to choose relevant features based on their importance for classification. The example below uses the chi-squared test to select the 20 best features. 

This works by selecting the features that are least likely to be independent of the class label (i.e. the features that are most likely to be relevant for classification).


```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_digits(return_X_y=True)
X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
```

<!--s-->

## Feature Selection | Random Forest

Another way is to reduce the feature space using modeling methods. The example below uses a random forest classifier to select the most important features. 

This works because the random forest model is selecting the features that are most likely to be important for classification. We'll cover random forests in more detail in a future lecture.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1010/1*R3oJiyaQwyLUyLZL-scDpw.png" width="300" style="display: block; margin: 0 auto; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Deniz Gunay, 2023</p>

</div>
<div class="c2" style = "width: 70%">

```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True)
X_new = SelectFromModel(RandomForestClassifier()).fit_transform(X, y)
```

</div>
</div>

<!--s-->

## A Note on OLAP / SnowFlake

You don't need to run these data preprocessing steps in your local environment. You can use OLAP / SnowFlake to run these steps in the cloud! 

Specifically, SnowFlake has something called a [Stored Procedure](https://docs.snowflake.com/en/developer-guide/stored-procedure/stored-procedures-overview), which is one way to run your Python preprocessing scripts in the cloud. Here is [example](https://docs.snowflake.com/en/developer-guide/stored-procedure/python/procedure-python-tabular-data) that uses Python.

A standard workflow may be to polish your preprocessing code in your local environment, and then run it in SnowFlake's scaled environment before your modeling exercises.

<!--s-->

<div class="header-slide">

# H.03 | preprocessing.py

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with preprocessing raw data to create a machine learning-ready dataset?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->









