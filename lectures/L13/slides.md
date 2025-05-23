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
  ## L.13 | Decision Trees & Ensemble Models

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

## Annoucements

- **H.04** will be released today at the end of class.
    - H.04 is due on Wednesday, May 21 at 11:59 PM.

- **Week 8** (next week) has three lectures:
    - Monday (May 19): Ford ITW
    - Wednesday (May 21): Kellogg L110
    - Thursday (May 22): Ford ITW

<!--s-->

## DIS Decision Making

You're a technical product manager at a large atheltic apparel company. Your team is working on a new app feature to predict the best time to run based on the user's location, weather conditions, and various biometric data. In short, this is a complex classification task based on many features with complex interactions.

Your lead scientist tells you that they are exploring two different modeling approaches.

1. **Decision-Tree Based Approach**: Using Random Forest or XGBoost.
2. **Deep Learning Approach**: Using a deep neural network with many layers.

Try to name at least one pro and one con for each approach. Which one do you want to start with? Why?

<!--s-->

<div class="header-slide">

# L.13 | Decision Trees & Ensemble Models

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Decision Trees
  2. Ensemble Models

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Decision Trees

</div>

<!--s-->

## Decision Trees | Overview

**Decision Trees** are a non-parametric supervised learning method used for classification and regression tasks. They are simple to **understand** and **interpret**, making them a popular choice in machine learning.

A decision tree is a tree-like structure where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome of the decision.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | ID3 Algorithm

The construction of a decision tree involves selecting the best feature to split the dataset at each node. One of the simplest algorithms for constructing categorical decision trees is the ID3 algorithm.

**ID3 Algorithm (Categorical Data)**:

1. Calculate the entropy of the target variable.
2. For each feature, calculate the information gained by splitting the data.
3. Select the feature with the highest information gain as the new node.

The ID3 algorithm recursively builds the decision tree by selecting the best feature to split the data at each node.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="40%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | Entropy

**Entropy** is a measure of the impurity or disorder in a dataset. The entropy of a dataset is 0 when all instances belong to the same class and is maximized when the instances are evenly distributed across all classes. Entropy is defined as: 

$$ H(S) = - \sum_{i=1}^{n} p_i \log_2(p_i) $$

Where:

- $H(S)$ is the entropy of the dataset $S$.
- $p_i$ is the proportion of instances in class $i$ in the dataset.
- $n$ is the number of classes in the dataset.

<img src="https://miro.medium.com/v2/resize:fit:565/1*M15RZMSk8nGEyOnD8haF-A.png" width="40%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Provost, Foster; Fawcett, Tom. </p>

<!--s-->

## Decision Trees | Information Gain

**Information Gain** is a measure of the reduction in entropy achieved by splitting the dataset on a particular feature. IG is the difference between the entropy of the parent dataset and the weighted sum of the entropies of the child datasets.

$$ IG(S, A) = H(S) - H(S|A)$$

Where:

- $IG(S, A)$ is the information gain of splitting the dataset $S$ on feature $A$.
- $H(S)$ is the entropy of the parent dataset.
- $H(S | A)$ is the weighted sum of the entropies of the child datasets.


<br><br>
<img src="https://miro.medium.com/max/954/0*EfweHd4gB5j6tbsS.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">KDNuggets</p>

<!--s-->

## Decision Trees | ID3 Pseudo-Code

```text
ID3 Algorithm:
1. If all instances in the dataset belong to the same class, return a leaf node with that class.
2. If the dataset is empty, return a leaf node with the most common class in the parent dataset.
3. Calculate the entropy of the dataset.
4. For each feature, calculate the information gain by splitting the dataset.
5. Select the feature with the highest information gain as the new node.
6. Recursively apply the ID3 algorithm to the child datasets.
```

Please note, ID3 works for **categorical** data. For **continuous** data, we can use the C4.5 algorithm, which is an extension of ID3 that supports continuous data.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | How to Split Continuous Data

For the curious, here is a simple approach to split continuous data with a decision tree. ID3 won't do this out of the box, but C4.5 does have an implementation for it.

1. Sort the dataset by the feature value.
2. Calculate the information gain for each split point.
3. Select the split point with the highest information gain as the new node.

```text
Given the following continuous feature:

    [0.7, 0.3, 0.4]

First you sort it: 

    [0.3, 0.4, 0.7]

Then you evaluate information gain for your target variable at every split:

    [0.3 | 0.4 , 0.7]

    [0.3, 0.4 | 0.7]
```

<!--s-->

## Decision Trees | Overfitting

**Overfitting** is a common issue with decision trees, where the model captures noise in the training data rather than the underlying patterns. Overfitting can lead to poor **generalization** performance on unseen data.

<img src="https://gregorygundersen.com/image/linoverfit/spline.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Gunderson 2020</p>


<!--s-->

## Decision Trees | Overfitting

**Overfitting** is a common issue with decision trees, where the model captures noise in the training data rather than the underlying patterns. Overfitting can lead to poor **generalization** performance on unseen data.

**Strategies to Prevent Overfitting in Decision Trees**:

1. **Pruning**: Remove branches that do not improve the model's performance on the validation data.
    - This is similar to L1 regularization in linear models!
2. **Minimum Samples per Leaf**: Set a minimum number of samples required to split a node.
3. **Maximum Depth**: Limit the depth of the decision tree.
4. **Maximum Features**: Limit the number of features considered for splitting.

<!--s-->


## Decision Trees | Pruning

**Pruning** is a technique used to reduce the size of a decision tree by removing branches that do not improve the model's performance on the validation data. Pruning helps prevent overfitting and improves the generalization performance of the model.

Practically, this is often done by growing the tree to its maximum size and then remove branches that do not improve the model's performance on the validation data. A loss function is used to evaluate the performance of the model on the validation data, and branches that do not improve the loss are pruned: 

$$ L(T) = \sum_{t=1}^{T} L(y_t, \widehat{y}_t) + \alpha |T| $$

Where:

- $L(T)$ is the loss function of the decision tree $T$.
- $L(y_t, \widehat{y}_t)$ is the loss function of the prediction $y_t$.
- $\alpha$ is the regularization parameter.
- $|T|$ is the number of nodes in the decision tree.


<!--s-->

## Decision Tree Algorithm Comparisons

| Algorithm | Data Types | Splitting Criteria |
|-----------|------------|--------------------|
| ID3       | Categorical | Entropy & Information Gain    |
| C4.5      | Categorical & Continuous | Entropy & Information Gain |
| CART      | Categorical & Continuous | Gini Impurity |


<!--s-->

## L.13 | Q.01

Which of the following decision tree algorithms is a reasonable choice for continuous data?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. ID3 <br>
&emsp;B. C4.5<br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.13 | Q.01" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Ensemble Models

</div>

<!--s-->

## Ensemble Models | Overview

**Ensemble Models** combine multiple individual models to improve predictive performance. The key idea behind ensemble models is that a group of weak learners can come together to form a strong learner. Ensemble models are widely used in practice due to their ability to reduce overfitting and improve generalization.

We will discuss two types of ensemble models:

1. **Bagging**: Bootstrap Aggregating
2. **Boosting**: Sequential Training

<!--s-->

## Ensemble Models | Bagging

**Bagging (Bootstrap Aggregating)** is an ensemble method that involves training multiple models on different subsets of the training data and aggregating their predictions. The key idea behind bagging is to reduce variance by averaging the predictions of multiple models.

**Intuition**: By training multiple models on different subsets of the data, bagging reduces the impact of outliers and noise in the training data. The final prediction is obtained by averaging the predictions of all models.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230731175958/Bagging-classifier.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Bagging | Example: Random Forests

**Random Forests** are an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. Random Forests use bagging to train multiple decision trees on different subsets of the data and aggregate their predictions.

**Key Features of Random Forests**:
- Each decision tree is trained on a random subset of the features.
- The final prediction is obtained by averaging the predictions of all decision trees.

Random Forests are widely used in practice due to their robustness, scalability, and ability to handle high-dimensional data.

<img src="https://tikz.net/janosh/random-forest.png" height="40%" style="margin: 0 auto; display: block; padding: 10">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting

**Boosting** is an ensemble method that involves training multiple models sequentially, where each model learns from the errors of its predecessor.

**Intuition**: By focusing on the misclassified instances in each iteration, boosting aims to improve the overall performance of the model. The final prediction is obtained by combining the predictions of all models.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting | Example: AdaBoost

**AdaBoost (Adaptive Boosting)** is a popular boosting algorithm. AdaBoost works by assigning weights to each instance in the training data and adjusting the weights based on the performance of the model.

**Key Features of AdaBoost**:

1. Train a weak learner on the training data.
2. Increase the weights of misclassified instances.
3. Train the next weak learner on the updated weights.
4. Repeat the process until a stopping criterion is met.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting | Example: Gradient Boosting

**Gradient Boosting** is another popular boosting algorithm. Gradient Boosting works by fitting a sequence of weak learners to the residuals of the previous model. This differs from AdaBoost, which focuses on the misclassified instances. A popular implementation of Gradient Boosting is **XGBoost** (❤️).

**Key Features of Gradient Boosting**:

1. Fit a weak learner to the training data.
2. Compute the residuals of the model.
3. Fit the next weak learner to the residuals.
4. Repeat the process until a stopping criterion is met.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200721214745/gradientboosting.PNG" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Running XGBoost in Python (XGBoost Package)

```python
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the Boston housing dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost regressor
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

# Fit the model
xg_reg.fit(X_train, y_train)

# Make predictions
preds = xg_reg.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
```

<!--s-->

## XGBoost | Original Paper

XGBoost was introduced by Tianqi Chen and Carlos Guestrin in 2016. The paper is titled "XGBoost: A Scalable Tree Boosting System" and is available [here](https://arxiv.org/pdf/1603.02754).

<img src="https://storage.googleapis.com/slide_assets/xgboost.png" width="70%" style="margin: 0 auto; display: block; border-radius: 10px;">

<!--s-->

## XGBoost | Scratch Implementation

<div style = "width: 100%;">


```python

"""
This script contains an implementation of the XGBoost algorithm.
The original XGBoost paper by Chen and Guestrin (2016): https://arxiv.org/pdf/1603.02754
Numpy code borrows heavily from https://randomrealizations.com/posts/xgboost-from-scratch/.
The code is written for educational purposes and is not optimized for performance.

Joshua D'Arcy, 2024.
"""

import math
import numpy as np 
from dataclasses import dataclass
from typing import Callable

@dataclass
class Parameters:
    learning_rate: float = 0.1
    max_depth: int = 5
    subsample: float = 0.8
    reg_lambda: float = 1.5
    gamma: float = 0.0
    min_child_weight: float = 25
    base_score: float = 0.0
    random_seed: int = 42

class XGBoost():
    
    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.base_prediction = self.params.base_score if self.params.base_score else 0.5
        self.rng = np.random.default_rng(seed=parameters.random_seed)
                
    def fit(self, X: np.ndarray, y: np.ndarray, objective: Callable, num_boost_round: int):

        # Initialize the base prediction.
        current_predictions = self.base_prediction * np.ones(shape=y.shape)

        # Initialize the list of boosters.
        self.boosters = []

        # Iterate over the number of boosting rounds.
        for i in range(num_boost_round):

            # Compute the first and second order gradients.
            first_order = objective.first_order(y, current_predictions)
            second_order = objective.second_order(y, current_predictions)

            # Get the sample indices (if subsampling is used).
            if self.params.subsample == 1.0:
                sample_idxs = np.arange(len(y))
            else:
                sample_idxs = self.rng.choice(len(y), size=math.floor(self.params.subsample*len(y)), replace=False)

            # Train a new Tree (booster) on the gradients and hessians.
            booster = Tree(X, first_order, second_order, self.params, sample_idxs)

            # Update the current predictions.
            current_predictions += self.params.learning_rate * booster.predict(X)

            # Append the new booster to the list of boosters.
            self.boosters.append(booster)
            
    def predict(self, X: np.ndarray):
        
        # Return the final prediction, based on the base prediction and the sum of the predictions of all boosters.
        summed_predictions = np.sum([booster.predict(X) for booster in self.boosters], axis=0)
        return (self.base_prediction + self.params.learning_rate * summed_predictions)

class Tree:
    def __init__(self, X: np.ndarray, first_order: np.ndarray, second_order: np.ndarray, parameters: Parameters, indices: np.ndarray, current_depth: int = 0):
        # Initialize the parameters.
        self.params = parameters
        self.X = X
        self.first_order = first_order
        self.second_order = second_order
        self.idxs = indices
        self.n = len(indices)
        self.c = X.shape[1]
        self.current_depth = current_depth

        # Equation (5) in the XGBoost paper.
        self.value = -first_order[indices].sum() / (second_order[indices].sum() + self.params.reg_lambda)

        # Set the initial best score to 0.
        self.best_score_so_far = 0.0

        # Initialize the left and right child nodes if max depth is not reached.
        if self.current_depth < self.params.max_depth:
            self.insert_child_nodes()

    def insert_child_nodes(self):
        # Find the best split for each feature.
        for i in range(self.c):
            self.find_split(i)

        # Check if the current node is a leaf node.
        if self.best_score_so_far == 0.0:
            return

        # Get x, the feature values for the current node.
        x = self.X[self.idxs, self.split_feature_idx]

        # Get the indices of the left and right child nodes.
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]

        # Create the left and right child nodes, incrementing depth.
        self.left = Tree(self.X, self.first_order, self.second_order, self.params, self.idxs[left_idx], self.current_depth + 1)
        self.right = Tree(self.X, self.first_order, self.second_order, self.params, self.idxs[right_idx], self.current_depth + 1)

    def find_split(self, feature_idx: int):

        # Get the feature values for the current node.
        x = self.X[self.idxs, feature_idx]

        # Sort the feature values.
        first = self.first_order[self.idxs]
        second = self.second_order[self.idxs]
        sort_idx = np.argsort(x)
        sort_first = first[sort_idx]
        sort_second = second[sort_idx]
        sort_x = x[sort_idx]

        # Initialize the sum of the first and second order gradients.
        sum_first = first.sum()
        sum_second = second.sum()
        sum_first_right = sum_first
        sum_second_right = sum_second
        sum_first_left = 0.0
        sum_second_left = 0.0

        for i in range(0, self.n - 1):

            # Get the first and second order gradients for the current split.
            first_i = sort_first[i]
            second_i = sort_second[i]
            x_i = sort_x[i]
            x_i_next = sort_x[i + 1]

            # Update the sum of the first and second order gradients.
            sum_first_left += first_i
            sum_first_right -= first_i
            sum_second_left += second_i
            sum_second_right -= second_i

            # Skip if the current split does not meet the minimum child weight requirement.
            if sum_second_left < self.params.min_child_weight or x_i == x_i_next:
                continue

            if sum_second_right < self.params.min_child_weight:
                break

            # Compute the gain of the current split, Equation (7) in the XGBoost paper.
            first_term = sum_first_left ** 2 / (sum_second_left + self.params.reg_lambda)
            second_term = sum_first_right ** 2 / (sum_second_right + self.params.reg_lambda)
            third_term = sum_first ** 2 / (sum_second + self.params.reg_lambda)
            gain = 0.5 * (first_term + second_term - third_term) - self.params.gamma / 2

            # Update the best split if the current split is better.
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2

    def predict(self, X: np.ndarray):
        # Iterate over each row in the input data and make a prediction.
        return np.array([self.predict_row(row) for row in X])

    def predict_row(self, row: np.ndarray):
        # Check if the current node is a leaf node.
        if self.best_score_so_far == 0.0:
            return self.value

        # If the current node is not a leaf node, then we need to find the child node.
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right

        # Recursively call the predict_row method on the child node.
        return child.predict_row(row)

```

</div>

<!--s-->

## L.13 | Q.02

Let's say that you train a sequence of models that learn from the mistakes of the predecessors. Instead of focusing on the misclassified instances (and weighting them more highly), you focus on improving the residuals. What algorithm is this most similar to?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
A. Bagging (e.g. Random Forest) <br>
B. Boosting (e.g. Adaboost) <br>
C. Boosting (e.g. Gradient Boosting) <br>
D. Stacking
</div>
</div>
<div class='c2' style = 'width: 40%;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.13 | Q.02" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>
<!--s-->

## Summary

In this lecture, we explored two fundamental concepts in supervised machine learning:

1. **Decision Trees**:
    - A non-parametric supervised learning method used for classification and regression tasks.
    - Can be constructed using the ID3 algorithm based on entropy and information gain.
    - Prone to overfitting, which can be mitigated using pruning and other strategies.

2. **Ensemble Models**:
    - Combine multiple individual models to improve predictive performance.
    - Include bagging, boosting, and stacking as popular ensemble methods.
    - Reduce overfitting and improve generalization by combining multiple models.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Decision Trees
  2. Ensemble Models

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# H.04 | Linear Regression

<span style="line-height: 1;">

### NUM_ATTEMPTS Updated to 4 

</span>

</div>

<!--s-->