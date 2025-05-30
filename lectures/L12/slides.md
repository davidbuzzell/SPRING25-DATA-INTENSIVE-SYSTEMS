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
  ## L.11 | Regression & Classification

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

- Exam Part I Graded
  - 2.5 points added to everyone's exam (+6.5%)
  - Exams have been released.

- Grading in MBAI 417

- H.04 will be released on Thursday (05.15.2025) and due on Monday (05.21.2025) @ 11:59PM.


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Linear Regression
  2. Logistic Regression

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Supervised Machine Learning

Supervised machine learning is a type of machine learning where the model is trained on a labeled dataset. Assume we have some features $X$ and a target variable $y$. The goal is to learn a mapping from $X$ to $y$.

$$ y = f(X) + \epsilon $$

Where $f(X)$ is the true relationship between the features and the target variable, and $\epsilon$ is a random error term. The goal of supervised machine learning is to estimate $f(X)$.

`$$ \widehat{y} = \widehat{f}(X) $$`

Where $\widehat{y}$ is the predicted value and $\widehat{f}(X)$ is the estimated function.

<!--s-->

## Supervised Machine Learning

To learn the mapping from $X$ to $y$, we need to define a model that can capture the relationship between the features and the target variable. The model is trained on a labeled dataset, where the features are the input and the target variable is the output.

- Splitting Data
  - Splitting data into training, validation, and testing sets.
- Linear Regression
  - Fundamental regression algorithm.
- Logistic Regression
  - Extension of linear regression for classification.

<!--s-->

## Splitting Data into Training, Validation, and Testing Sets

Splitting data into training, validation, and testing sets is crucial for model evaluation and selection.

- **Training Set**: Used for fitting the model.
- **Validation Set**: Used for parameter tuning and model selection.
- **Test Set**: Used to evaluate the model performance.

A good, general rule of thumb is to split the data into 70% training, 15% validation, and 15% testing. In practice, k-fold cross-validation is often used to maximize the use of data. We will discuss k-folds in a future lecture.

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

<!--s-->

## Methods of Splitting

<div class = "col-wrapper">

<div class="c1" style = "width: 60%; font-size: 0.8em;">

### Random Split
Shuffles the data to avoid bias. Important when your dataset is ordered.

### Stratified Split
Used with imbalanced data to ensure each set reflects the overall distribution of the target variable. Important when your dataset has a class imbalance.

### Time-Based Split
Used for time series data to ensure the model is evaluated on future data. Important when your dataset is time-dependent.

### Group-Based Split
Used when data points are not independent, such as in medical studies. Important when your dataset has groups of related data points.

</div>

<div class="c2 col-centered" style = "width: 40%;">

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

</div>

<!--s-->

## Why Split on Groups?

Imagine we're working with a medical dataset aimed at predicting the likelihood of a patient having a particular disease based on various features, such as age, blood pressure, cholesterol levels, etc.

### Dataset

- **Patient A:** 
  - Visit 1: Age 50, Blood Pressure 130/85, Cholesterol 210
  - Visit 2: Age 51, Blood Pressure 135/88, Cholesterol 215

- **Patient B:**
  - Visit 1: Age 60, Blood Pressure 140/90, Cholesterol 225
  - Visit 2: Age 61, Blood Pressure 145/92, Cholesterol 230

<!--s-->

## Incorrect Splitting

- **Training Set:**
  - Patient A, Visit 1
  - Patient B, Visit 1

- **Testing Set:**
  - Patient A, Visit 2
  - Patient B, Visit 2

In this splitting scenario, the model could learn specific patterns from Patient A and Patient B in the training set and then simply recall them in the testing set. Since it has already seen data from these patients, even with slightly different features, **it may perform well without actually generalizing to unseen patients**.

<!--s-->

## Correct Splitting

- **Training Set:**
  - Patient A, Visit 1
  - Patient A, Visit 2

- **Testing Set:**
  - Patient B, Visit 1
  - Patient B, Visit 2

In these cases, the model does not have prior exposure to the patients in the testing set, ensuring an unbiased evaluation of its performance. It will need to apply its learning to truly "new" data, similar to real-world scenarios where new patients must be diagnosed based on features the model has learned from different patients.

<!--s-->

## Key Takeaways for Group-Based Splitting

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Avoiding Data Leakage

By ensuring that the model does not see any data from the testing set during training, we prevent data leakage, which can lead to overly optimistic performance metrics.

</div>
<div class="c2" style = "width: 50%">

### Real-World Applicability

This method simulates real-world scenarios where the model will encounter new groups (in the example before, patients) or data points that it has never seen before.

</div>
</div>

<!--s-->

<div class="header-slide">

# Linear Regression

</div>

<!--s-->

## Linear Regression | Concept

Linear regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. The components to perform linear regression: 

$$ \widehat{y} = X\beta $$

Where $ \widehat{y} $ is the predicted value, $ X $ is the feature matrix, and $ \beta $ is the coefficient vector. The goal is to find the coefficients that minimize the error between the predicted value and the actual value.

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="600" style="margin: 0 auto; display: block;">

<!--s-->

## Linear Regression | Cost Function

The objective of linear regression is to minimize the cost function $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

Where $ \widehat{y} = X\beta $ is the prediction. This is most easily solved by finding the normal equation solution:

$$ \beta = (X^T X)^{-1} X^T y $$

The normal equation is derived by setting the gradient of $J(\beta) $ to zero. This is a closed-form solution that can be computed directly.

</div>
<div class="c2" style = "width: 50%">

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)
```

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="100%">

</div>
</div>

<!--s-->

## Linear Regression | Normal Equation Notes

### Adding a Bias Term

Practically, if we want to include a bias term in the model, we can add a column of ones to the feature matrix $ X $.

$$ \widehat{y} = X\beta $$

### Gradient Descent

For large datasets, the normal equation can be computationally expensive. Instead, we can use gradient descent to minimize the cost function iteratively. We'll talk about gradient descent within the context of logistic regression later today.

<!--s-->

## Linear Regression | Regression Model Evaluation

To evaluate a regression model, we can use metrics such as mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE), and R-squared.

<div style = "font-size: 0.8em;">

| Metric | Formula | Notes |
| --- | --- | --- |
| Mean Squared Error (MSE) | `$\frac{1}{m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2$` | Punishes large errors more than small errors. |
| Mean Absolute Error (MAE) | `$\frac{1}{m} \sum_{i=1}^m \|\widehat{y}_i - y_i\|$` | Less sensitive to outliers than MSE. |
| Mean Absolute Percentage Error (MAPE) | `$\frac{1}{m} \sum_{i=1}^m \left \| \frac{\widehat{y}_i - y_i}{y_i} \right\| \times 100$` | Useful for comparing models with different scales. |
| R-squared | `$1 - \frac{\sum(\widehat{y}_i - y_i)^2}{\sum(\bar{y} - y_i)^2}$` | Proportion of the variance in the dependent variable that is predictable from the independent variables. |

</div>

<!--s-->

## Linear Regression | Pros and Cons

### Pros

- Simple and easy to understand.
- Fast to train.
- Provides a good, very interpretable baseline model.

### Cons

- Assumes a linear relationship between the features and the target variable.
- Sensitive to outliers.

<!--s-->

## Linear Regression | A Brief Note on Regularization

<div style = "font-size: 0.8em;">
Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. The two most common types of regularization are L1 (Lasso) and L2 (Ridge) regularization.

Recall that the cost function for linear regression is:

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

**L1 Regularization**: Adds the absolute value of the coefficients to the cost function. This effectively performs feature selection by pushing some coefficients towards zero.

$$ J(\beta) = J(\beta) + \lambda \sum_{j=1}^n |\beta_j| $$

**L2 Regularization**: Adds the square of the coefficients to the cost function. This shrinks the coefficients, but does not set them to zero. This is useful when all features are assumed to be relevant.

$$ J(\beta) = J(\beta) + \lambda \sum_{j=1}^n \beta_j^2 $$
</div>

<!--s-->

## Linear Regression | A Brief Note on P-Values

P-values are used to test the null hypothesis that the coefficient is equal to zero. A small p-value (typically < 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. 

In linear regression, the null hypothesis is that the coefficient is equal to zero, meaning that the feature does not have a significant effect on the target variable. The alternative hypothesis is that the coefficient is not equal to zero, meaning that the feature does have a significant effect on the target variable.

<!--s-->

## L.07 | Q.01

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

When is L2 regularization (Ridge) preferred over L1 regularization (Lasso)?

A. When all features are assumed to be relevant.<br>
B. When some features are assumed to be irrelevant.

</div>

<div class="c2" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.01" width="100%" height="100%" style="border-radius: 10px;"></iframe>

</div>

</div>

<!--s-->

<div class="header-slide">

# Logistic Regression

</div>

<!--s-->

## Logistic Regression | Concept

Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function.

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*G3imr4PVeU1SPSsZLW9ghA.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Joshi, 2019</span>

<!--s-->

## Logistic Regression | Formula

This model is based on the sigmoid function $\sigma(z)$:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Where 

$$ z = X\beta $$

Note that $\sigma(z)$ is the probability that the dependent variable is 1 given the input $X$. Consider the similar form of the linear regression model:

$$ \widehat{y} = X\beta $$

The key difference is that the output of logistic regression is passed through the sigmoid function to obtain a value between 0 and 1, which can be interpreted as a probability. This works because the sigmoid function maps any real number to the range [0, 1]. While linear regression predicts the value of the dependent variable, logistic regression predicts the probability that the dependent variable is 1. 

<!--s-->

## Logistic Regression | No Closed-Form Solution

In linear regression, we can calculate the optimal coefficients $\beta$ directly. However, in logistic regression, we cannot do this because the sigmoid function is non-linear. This means that there is no closed-form solution for logistic regression.

Instead, we use gradient descent to minimize the cost function. Gradient descent is an optimization algorithm that iteratively updates the parameters to minimize the cost function, and forms the basis of many machine learning algorithms.

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">machinelearningspace.com (2013)</p>

<!--s-->

## Logistic Regression | Cost Function

The cost function used in logistic regression is the binary cross-entropy loss:

$$ J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\widehat{y}_i) + (1 - y_i) \log(1 - \widehat{y}_i)] $$

$$ \widehat{y} = \sigma(X\beta) $$

Let's make sure we understand the intuition behind the cost function $J(\beta)$.

If the true label ($y$) is 1, we want the predicted probability ($\widehat{y}$) to be close to 1. If the true label ($y$) is 0, we want the predicted probability ($\widehat{y}$) to be close to 0. The cost goes up as the predicted probability diverges from the true label.

<!--s-->

## Logistic Regression | Gradient Descent

To minimize $ J(\beta) $, we update $ \beta $ iteratively using the gradient of $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

$$ \beta := \beta - \alpha \frac{\partial J}{\partial \beta} $$

Where $ \alpha $ is the learning rate, and the gradient $ \frac{\partial J}{\partial \beta} $ is:

$$ \frac{\partial J}{\partial \beta} = \frac{1}{m} X^T (\sigma(X\beta) - y) $$

Where $ \sigma(X\beta) $ is the predicted probability, $ y $ is the true label, $ X $ is the feature matrix, $ m $ is the number of instances, $ \beta $ is the coefficient vector, and $ \alpha $ is the learning rate.

This is a simple concept that forms the basis of many gradient-based optimization algorithms, and is widely used in deep learning. 

Similar to linear regression -- if we want to include a bias term, we can add a column of ones to the feature matrix $ X $.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">machinelearningspace.com (2013)</p>

</div>
</div>

<!--s-->

## L.07 | Q.02

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Okay, so let's walk through an example. Suppose you have already done the following: 

1. Obtained the current prediction ($\widehat{y}$) with $ \sigma(X\beta) $.
2. Calculated the gradient $ \frac{\partial J}{\partial \beta} $.

What do you do next?


</div>
<div class="c2" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.02" width="100%" height="100%" style="border-radius: 10px;"></iframe>

</div>
</div>

<!--s-->

## Logistic Regression | Classifier

Once we have the optimal coefficients, we can use the logistic function to predict the probability that the dependent variable is 1. 

We can then use a threshold to classify the instance as 0 or 1 (usually 0.5). The following code snippet shows how to use the scikit-learn library to fit a logistic regression model and make predictions.


```python
from sklearn.linear_model import LogisticRegression

X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
logistic_regression_model.predict([[3.5, 3.5]])
```

```python
array([[0.3361201, 0.6638799]])
```

<!--s-->

## L.07 | Q.03

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Our logistic regression model was trained with $X = [[1, 2], [2, 3], [3, 4], [4, 5]]$ and $y = [0, 0, 1, 1]$. We then made a prediction for the point $[3.5, 3.5]$.

What does this output represent?

```python
array([[0.3361201, 0.6638799]])
```

</div>
<div class="c2" style = "width: 50%">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.03" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

## Logistic Regression | Classification Model Evaluation

To evaluate a binary classification model like this, we can use metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

| Metric | Formula | Notes |
| --- | --- | --- | 
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Easy to interpret but flawed.
| Precision | $\frac{TP}{TP + FP}$ | Useful when the cost of false positives is high. |
| Recall | $\frac{TP}{TP + FN}$ | Useful when the cost of false negatives is high. |
| F1 Score | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Harmonic mean of precision and recall. | 
| ROC-AUC | Area under the ROC curve. | Useful for imbalanced datasets. |

<!--s-->

## Logistic Regression | Pros and Cons

### Pros

- Simple and easy to understand.
- Fast to train.
- Provides a good, very interpretable baseline model.

### Cons

- Assumes a linear relationship between the features and the target variable.
- Sensitive to outliers.
- Does not handle non-linear relationships well.

<!--s-->

## Summary

- We discussed the importance of splitting data into training, validation, and test sets.
- We delved into Linear Regression and Logistic Regression with Gradient Descent, exploring practical implementations and theoretical foundations.
- Understanding these foundational concepts is crucial for advanced machine learning and model fine-tuning! 

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following methods:


  1. Linear Regression
  2. Logistic Regression

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->