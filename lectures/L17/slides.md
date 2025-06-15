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
  ## L.17 | Training the *Best* Model

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

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with cross-validation & hyperparameter tuning?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Topic Poll
  ## What topic do you want to learn about in the next (final) lecture?

  A. MLOps<br>
  B. Model Deployment<br>
  C. Explainable AI<br>

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Topic Poll" width = "100%" height = "100%"></iframe>
</div>

<!--s-->

## Agenda

### ROC-AUC

Reciever Operating Characteristic - Area Under the Curve (ROC-AUC) is a performance measurement for classification problems at various threshold settings.

### k-fold cross-validation

k-fold cross-validation is a technique to use all of your data for training and validation, which can lead to more accurate estimates of model performance.

### Hyperparameter tuning

Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance.

<!--s-->

<div class="header-slide">

# ROC-AUC

</div>

<!--s-->

## ROC-AUC

- **Receiver Operating Characteristic (ROC) Curve**: A graphical representation of the performance of a binary classifier system as its discrimination threshold is varied.

- **Area Under the Curve (AUC)**: The area under the ROC curve, which quantifies the classifier’s ability to distinguish between classes.

<img src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/64760779d5dc484958a3f917_classification_metrics_017-min.png" width="400" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Source: Evidently AI</p>

<!--s-->

## ROC-AUC | Key Concepts

<div style="font-size: 0.7em;">

**True Positive Rate (TPR)**: The proportion of actual positive cases that are correctly identified by the classifier.<br>
**False Positive Rate (FPR)**: The proportion of actual negative cases that are incorrectly identified as positive by the classifier.<br>
**ROC**: When the FPR is plotted against the TPR for each binary classification threshold, we obtain the ROC curve.
</div>
<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*CQ-1ceyX80EE0a_s3SwvgQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## ROC-AUC | Key Concepts

<img src="https://miro.medium.com/v2/resize:fit:4512/format:webp/1*zNtuQziwUKkGUxhG0Go5kA.png" width="100%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## L.17 | Q.01

I have a binary classifier that predicts whether a patient has a disease. The ROC-AUC of the classifier is 0.4. What does this mean?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. The classifier is worse than random. <br>
&emsp;B. The classifier is random. <br>
&emsp;C. The classifier is better than random. <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.17 | Q.01" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# K-Fold Cross-Validation

</div>

<!--s-->

## K-Fold Cross-Validation

K-Fold Cross-Validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the data into K equal-sized folds, training the model on K-1 folds, and evaluating the model on the remaining fold. This process is repeated K times, with each fold serving as the validation set once.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*AAwIlHM8TpAVe4l2FihNUQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Patro 2021</span>

<!--s-->

## K-Fold Cross-Validation | Advantages

When implemented correctly, K-Fold Cross-Validation has several advantages:

- **Better Use of Data**: K-Fold Cross-Validation uses all the data for training and validation, which can lead to more accurate estimates of model performance.

- **Reduced Variance**: By averaging the results of K different validation sets, K-Fold Cross-Validation can reduce the variance of the model evaluation.

- **Model Selection**: K-Fold Cross-Validation can be used to select the best model hyperparameters.

<!--s-->

<div class="header-slide">

# Hyperparameter Tuning Strategies

</div>

<!--s-->

## Hyperparameter Tuning Strategies

Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance.

We will cover 3 strategies in detail:

1. Grid Search
2. Random Search
3. Hyperband

<!--s-->

## Goal

The goal of hyperparameter tuning is to find the optimal set of hyperparameters that minimize the model's loss on a validation set.

`$$ \text{argmin}_{\theta} \sum_{i=1}^{n} L(y_i, f(x_i, \theta)) $$`

where:

- $ \theta $ is the hyperparameter vector
- $ y_i $ is the true label
- $ f(x_i, \theta) $ is the predicted label
- $ L $ is the loss function
- $ n $ is the number of samples

<!--s-->

## Grid Search

Grid search is a brute-force approach to hyperparameter tuning. It involves defining a grid of hyperparameter values and evaluating the model's performance for each combination of hyperparameters.

```python
for learning_rate in [0.01, 0.1, 1]:
    for batch_size in [16, 32, 64]:
        model = create_model(learning_rate, batch_size)
        model.fit(X_train, y_train)
        score = model.evaluate(X_val, y_val)
        store_results(learning_rate, batch_size, score)
```

<!--s-->

## Grid Search

| Pros | Cons |
| --- | --- |
| Simple to implement | Computationally expensive |
| Easy to understand | Not suitable for large hyperparameter spaces |
| Guarantees finding the optimal hyperparameters | Can be inefficient if the grid is not well-defined |

<!--s-->

## Random Search

Random search is a more efficient alternative to grid search. Instead of evaluating all combinations of hyperparameters, it randomly samples a fixed number of combinations and evaluates their performance.

Random search has been shown to be more efficient than grid search in practice, especially for large hyperparameter spaces [[citation]](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

```python
# Create random hyperparameter combinations.
hyperparameter_space = {
    'learning_rate': [0.01, 0.1, 1],
    'batch_size': [16, 32, 64, 128, 256]
}

random_combinations = random.sample(list(itertools.product(*hyperparameter_space.values())), k=10)

for learning_rate, batch_size in random_combinations:
    model = create_model(learning_rate, batch_size)
    model.fit(X_train, y_train)
    score = model.evaluate(X_val, y_val)
    store_results(learning_rate, batch_size, score)
```


<!--s-->

## Random Search


| Pros | Cons |
| --- | --- |
| More efficient than grid search | No guarantee of finding the optimal hyperparameters |
| Can be parallelized | Randomness can lead to inconsistent results |
| Suitable for large hyperparameter spaces | Requires careful selection of the number of samples |
| Can find good hyperparameters even with a small number of evaluations | |

<!--s-->

## Hyperband

Hyperband is a more advanced hyperparameter tuning algorithm that combines random search with early stopping. It is based on the idea of running multiple random search trials in parallel and stopping the least promising trials early.

This smart resource allocation strategy allows Hyperband to find good hyperparameters with fewer evaluations than random search. The authors indicate it is also superior to Bayesian optimization [[citation]](https://arxiv.org/abs/1603.06560).


```python

from keras_tuner import Hyperband

tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=50,
    factor=3,
    directory='my_dir',
    project_name='helloworld',

)

tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
```

<!--s-->

## Hyperband Pseudocode

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Li 2018</p>
</div>

<!--s-->

## Hyperband Explanation

$n$ is the number of configurations to evaluate in each bracket. It ensures that the total budget $B$ is distributed across all brackets, taking into account the reduction factor $η$ and the number of brackets $s$.

**Number of configurations (n):**
$$ n = \left\lceil \frac{B}{R} \cdot \frac{\eta^s}{s + 1} \right\rceil $$

Where:
  - $B$ is the total budget
  - $R$ is the resources allocated to each configuration
  - $s$ is the number of brackets
  - $η$ is the reduction factor (e.g., 3)

<!--s-->

## Hyperband Explanation

This formula determines the initial amount of resources allocated to each configuration in a given bracket. As $s$ decreases, the resources per configuration increase.

**Resources per configuration (r):**
$$ r = R \cdot \eta^{-s} $$

Where:
  - $R$ is the total budget
  - $s$ is the number of brackets
  - $η$ is the reduction factor (e.g., 3)
  
<!--s-->

## Hyperband Example

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey; margin: 0px;'>Li 2018</p>
</div>

</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center; width: 70%'>
   <img src='https://storage.googleapis.com/slide_assets/hyperband_from_pseudo.png' style='border-radius: 10px;'>
</div>

</div>
</div>

<!--s-->

## Hyperband

| Pros | Cons |
| --- | --- |
| Efficient resource allocation | Requires hyperparameter tuning |
| Combines exploration and exploitation | Performance depends on the choice of initial configurations |
| Suitable for large hyperparameter spaces | May not always find the global optimum |
| Can find good hyperparameters with fewer evaluations compared to random search | |

<!--s-->

## Other Hyperparameter Tuning Strategies


| Strategy | Description |
| --- | --- |
| Bayesian Optimization | Uses probabilistic models to find the optimal hyperparameters by balancing exploration and exploitation. |
| Genetic Algorithms | Uses evolutionary algorithms to optimize hyperparameters by mimicking natural selection processes. |
| Tree-structured Parzen Estimator (TPE) | A Bayesian optimization method that models the distribution of good and bad hyperparameters separately. |
| Reinforcement Learning | Uses reinforcement learning algorithms to optimize hyperparameters by treating the tuning process as a sequential decision-making problem.

<!--s-->

<div class="header-slide">

# Hyperparameter Tuning Demo

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with cross-validation & hyperparameter tuning?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->
