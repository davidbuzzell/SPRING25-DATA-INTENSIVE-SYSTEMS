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
  ## L.14 | Time Series Modeling

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
  ## Please check in by entering the provided code.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- H.04 is due Wednesday at 11:59PM.

- This week has three lectures:
    - Monday (May 19): Ford ITW
    - Wednesday (May 21): Kellogg L110
    - Thursday (May 22): Ford ITW

- H.05 will be released next Thursday.

<!--s-->

<div class="header-slide">

# L.14 | Time Series Modeling

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Seasonal Decomposition
  2. Stationarity & Differencing
  3. Autocorrelation
  4. Autoregressive Models
  5. Modern Forecasting Models

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Agenda

Today we're going to talk about time series analysis, specifically building an intution for forecasting models. We'll cover the following topics:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Understanding Time Series Data
- Seasonal Decomposition
- Stationarity & Differencing
- Autocorrelation

### Time Series Forecasting
- Autoregressive Models
- XGBoost
- D-Linear & N-Linear Models
- TiDE

### Evaluation
- Walk-Forward Validation
- Evaluation Metrics

</div>
<div class="c2 col-centered" style = "width: 50%">

<div>
<img src="https://storage.googleapis.com/slide_assets/forecast_lokad.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Lokad 2016</span>
</div>
</div>
</div>

<!--s-->

<div class="header-slide">

# Understanding Time Series Data

</div>

<!--s-->

## Understanding Time Series | Seasonal Decomposition

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin: 0; padding: 0;">

### Trend
The long-term movement of a time series. That represents the general direction in which the data is moving over time.

### Seasonality
The periodic fluctuations in a time series that occur at regular intervals. For example, sales data may exhibit seasonality if sales increase during the holiday season.

### Residuals
Noise in a time series that cannot be explained by the trend or seasonality.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>

<img src="https://sthalles.github.io/assets/time-series-decomposition/complete-seasonality-plot-additive.png" width="400" style="margin: 0; padding: 0; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Thalles 2019</span>

</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Seasonal Decomposition

Seasonal Decomposition is a technique used to separate a time series into its trend, seasonal, and residual components. Seasonal decomposition can help identify patterns in the time series data and make it easier to model. It can be viewed as a form of feature engineering.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Additive Seasonal Decomposition
The seasonal component is added to the trend and residual components.

$$ X_t = T_t + S_t + R_t $$

### Multiplicative Seasonal Decomposition

The seasonal component is multiplied by the trend and residual components.
$$ X_t = T_t \times S_t \times R_t $$

</div>

<div class="c2" style = "width: 50%; margin-top: 8%;">
<div>
<img src="https://sthalles.github.io/assets/time-series-decomposition/complete-seasonality-plot-additive.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Thalles 2019</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Stationarity

A time series is said to be **stationary** if its statistical properties such as mean, variance, and autocorrelation do not change over time. Many forecasting methods assume that the time series is stationary. The **Augmented Dickey-Fuller Test (ADF)** is a statistical test that can be used to test for stationarity.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Strict Stationarity
The joint distribution of any subset of time series observations is independent of time. This is a strong assumption that is rarely met in practice.

### Trend Stationarity
The mean of the time series is constant over time. This is a weaker form of stationarity that is more commonly used in practice.

</div>
<div class="c2" style = "width: 50%">

<div>
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/Stationarycomparison.png" style="margin: 0; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Wikipedia 2024</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Differencing

**Differencing** is a technique used to make a time series **stationary** by computing the difference between consecutive observations. Differencing can help remove trends and seasonality from a time series.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ Y_t' = Y_t - Y_{t-1} $$

Where:
- $Y_t$ is the observation at time $t$.
- $Y_t'$ is the differenced observation at time $t$.

</div>
<div class="c2" style = "width: 50%">

<img src="https://storage.googleapis.com/blogs-images-new/ciscoblogs/1/2020/03/0e3efdd8-differencing.png" width="400" style="margin: 0 auto; display: block; border-radius: 10px;">
<span style="font-size: 0.6em; text-align: center; display: block; color: grey;">Wise, 2020</span>

</div>
</div>

<!--s-->

## Understanding Time Series | Autocorrelation

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. 

$$ \text{Corr}(X_t, X_{t-k}) $$


### Partial Autocorrelation
A measure of the correlation between a time series and a lagged version of itself, controlling for the values of the time series at all shorter lags.

$$ \text{Corr}(X_t, X_{t-k} | X_{t-1}, X_{t-2}, \ldots, X_{t-k+1}) $$

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src = "https://i.makeagif.com/media/3-17-2017/CYdNJ7.gif" width="100%" style="margin: 0 auto; display: block; border-radius: 10px;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey; padding-top: 0.5em;">@osama063, 2016</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Autocorrelation

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. 

$$ \text{Corr}(X_t, X_{t-k}) $$

### Partial Autocorrelation
A measure of the correlation between a time series and a lagged version of itself, controlling for the values of the time series at all shorter lags.

$$ \text{Corr}(X_t, X_{t-k} | X_{t-1}, X_{t-2}, \ldots, X_{t-k+1}) $$

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://storage.googleapis.com/cs326-bucket/lecture_13/observed.png" width="100%" style="margin: 0 auto; display: block;">
<img src="https://storage.googleapis.com/cs326-bucket/lecture_13/auto2.png" width="100%" style="margin: 0 auto; display: block;">
</div>
</div>
</div>


<!--s-->

## Understanding Time Series | Checkpoint TLDR;

### Seasonal Decomposition
A technique used to separate a time series into its trend, seasonal, and residual components.

### Stationarity
A time series is said to be stationary if its basic properties do not change over time.

### Differencing
A technique used to make a time series stationary by computing the difference between consecutive observations.

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. Partial autocorrelation controls for the values of the time series at all shorter lags.


<!--s-->

<div class="header-slide">

# Time Series Forecasting

</div>

<!--s-->

## Time Series Forecasting | Introduction

Time series forecasting is the process of predicting future values based on past observations. Time series forecasting is used in a wide range of applications, such as sales forecasting, weather forecasting, and stock price prediction. 

The **ARIMA** (Autoregressive Integrated Moving Average) model is a popular time series forecasting model that combines autoregressive, moving average, and differencing components. Let's build an intution for the AR (Autoregressive) model.

<!--s-->

## Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(1) $$ X_t = \phi_1 X_{t-1} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $\phi_1$ is a learnable parameter of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(p) $$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $p$ is the number of lag observations included in the model.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the parameters of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used for capturing the periodic structure of the time series.

$$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

<iframe width = "100%" height = "70%" src="https://storage.googleapis.com/cs326-bucket/lecture_13/ARIMA_1_2.html" title="scatter_plot"></iframe>

<!--s-->

<div class="header-slide">

# Re-Appropriated ML for Time Series

</div>

<!--s-->

## XGBoost

XGBoost is a popular machine learning algorithm that can be used for time series forecasting. It is an ensemble learning method that combines the predictions of multiple weak learners (typically decision trees) to produce a strong learner.

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

XGBoost is not a time series model per se, but it can be adapted for time series forecasting by creating features from the time series data. This involves generating lagged features, rolling statistics, and other relevant features that capture the temporal patterns in the data. 

Then XGBoost is trained on these features to make predictions. This is similar to the Autoregressive models we discussed earlier, but with the added flexibility and power of gradient boosting.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://flower.ai/static/images/blog/content/2023-11-29-xgboost-pipeline.jpg' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0;'>Gao 2023</p>
</div>

</div>
</div>

<!--s-->

<div class="header-slide">

# Modern Forecasting with Deep Learning

</div>

<!--s-->

## Vanilla Linear Model

In 2022, Zeng et al proposed a simple linear model. This model is a straightforward approach to time series forecasting, where the future value is predicted as a linear combination of past values. Two variants of the linear model (N and D) are used to compare the performance of existing Transformer-based models at the time.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_model.png' style='border-radius: 10px; width: 80%;'>
   <p style='font-size: 0.6em; color: grey; margin: 0'>Zeng 2022</p>
</div>

<!--s-->

## D-Linear Model
[[original_paper](https://arxiv.org/pdf/2205.13504)]

The D-Linear model is a variant of the linear model that incorporates a trend component. The D-Linear model decomposes a raw data input into a trend component by a moving average kernel and a remainder (seasonal) component. Then, two linear layers are applied to each component, and we sum up the two features to get the final prediction.

D-Linear models are simple and effective, and should be considered when you have a strong trend component in your time series data.

<div style='text-align: center;'>
   <img src='https://images.squarespace-cdn.com/content/v1/678a4c72a9ba99192a50b3fb/b2ba5424-af32-4a97-b1c1-1015c4860c7c/decomp.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Davies</p>
</div>

<!--s-->

## N-Linear Model
[[original_paper](https://arxiv.org/pdf/2205.13504)]

The N-Linear model is another variant of the linear model that handles a distribution shift between training and testing sets. NLinear subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction.

N-Linear models are simple and effective, and should be considered when you have a distribution shift between training and testing sets.

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/linear_distribution_shift.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Zeng 2022</p>
</div>

<!--s-->

## TiDE
[[original_paper](https://arxiv.org/pdf/2304.08424)]

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

TiDE (Time-series Dense Encoder) is a deep learning model that utilizes a dense encoder architecture specifically designed for time series data. It captures complex patterns in the data and can be used for both univariate and multivariate time series forecasting. The model is designed to be efficient and can handle large datasets with high dimensionality.

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/tide.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Das 2024</p>
</div>

</div>
</div>

<!--s-->

<div class="header-slide">

# Forecast Evaluation

</div>

<!--s-->

## Walk-Forward Validation

In walk-forward validation, the model is trained on historical data and then used to make predictions on future data. The model is then retrained on the updated historical data and used to make predictions on the next future data point. This process is repeated until all future data points have been predicted.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Train / Validate Period
The historical data used to train and validate the time series model.

### Test Period
The future data used to evaluate the generalization performance of the time series model.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://www.researchgate.net/profile/Vasco-Leal/publication/341679308/figure/fig14/AS:895809256169474@1590588978362/Illustrative-diagram-of-the-walk-forward-validation-Taken-from-5.ppm" width="100%" style="margin: 0 auto; display: block;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey;">Leal, 2019</span>

</div>
</div>

<!--s-->

## Walk-Forward Validation

In walk-forward validation, the model is trained on historical data and then used to make predictions on future data. The model is then retrained on the updated historical data and used to make predictions on the next future data point. This process is repeated until all future data points have been predicted.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Train / Validate Period
The historical data used to train and validate the time series model.

### Test Period
The future data used to evaluate the generalization performance of the time series model.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://www.researchgate.net/publication/250036526/figure/fig2/AS:669330027081729@1536592122436/In-sample-and-out-of-sample-forecasting-for-DJUI-data.png" width="100%" style="margin: 0 auto; display: block;"> 
<span style="font-size: 0.5em; text-align: center; display: block; color: grey;">Karaman, 2005</span>

</div>
</div>

<!--s-->

## Evaluation Metrics

<div style = "font-size: 0.8em">

**Mean Absolute Error (MAE)**: The average of the absolute errors between the predicted and actual values.

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

**Mean Squared Error (MSE)**: The average of the squared errors between the predicted and actual values.

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Root Mean Squared Error (RMSE)**: The square root of the average of the squared errors between the predicted and actual values.

$$ RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } $$

**Mean Absolute Percentage Error (MAPE)**: The average of the absolute percentage errors between the predicted and actual values.

$$ MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\% $$

</div>

<!--s-->

<div class="header-slide">

# Wrapping Up

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Seasonal Decomposition
  2. Stationarity & Differencing
  3. Autocorrelation
  4. Autoregressive Models
  5. Modern Forecasting Models

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->