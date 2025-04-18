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
  ## L.06 | OLAP + EDA II

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

- H.02 is due tonight @ 11:59PM.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **EDA** concepts such as:

- Variance, Covariance, and Correlation
- Association Analysis

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Variance, Covariance, and Correlation

</div>

<!--s-->

## Descriptive EDA | Examples

- **Central tendency**
    - Mean, Median, Mode
- **Spread**
    - Range, Variance, interquartile range (IQR)

<!--s-->

## Central Tendency

- **Mean**: The average of the data. 

    - $ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $

    - <span class="code-span">np.mean(data)</span>

- **Median**: The middle value of the data, when sorted.

    - [1, 2, **4**, 5, 6]

    - <span class="code-span">np.median(data)</span>

- **Mode**: The most frequent value in the data.

    ```python
    from scipy.stats import mode
    data = np.random.normal(0, 1, 1000)
    mode(data)
    ```

<!--s-->

## Spread

- **Range**: The difference between the maximum and minimum values in the data.
    
    - <span class="code-span">np.max(data) - np.min(data)</span>

- **Variance**: The average of the squared differences from the mean.

    - $ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $

    - <span class="code-span">np.var(data)</span>

- **Standard Deviation**: The square root of the variance.

    - `$ \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2} $`
    - <span class="code-span">np.std(data)</span>

- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles.
    - <span class="code-span">np.percentile(data, 75) - np.percentile(data, 25)</span>

<!--s-->

## Correlation | Quantitative Measurement via Covariance

**Covariance** is a measure of how much two random variables vary together. The covariance between two variables \(X\) and \(Y\) can be defined as:

$$ \text{cov}(X, Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n} $$

Where:
- $x_i$ and $y_i$ are the individual data points of variables $X$ and $Y$.
- $\bar{x}$ and $\bar{y}$ are the means of variables $X$ and $Y$.
- $n$ is the number of data points.

<!--s-->

## Correlation | Interpreting Covariance

When the covariance is positive, it means that the two variables are moving in the same direction. When the covariance is negative, it means that the two variables are moving in opposite directions.

**But** size of the covariance is not standardized, so it is difficult to interpret the strength of the relationship. Consider the following example:

**Case 1:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[50, 60, 70, 80, 90]</span>

**Case 2:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[500, 600, 700, 800, 900]</span>

Covariance will be different in these cases, but the relationship is the same!

<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson correlation coefficient, denoted by $\(r\)$, is a measure of the linear correlation between two variables. It ranges from -1 to 1, and so it is a **standardized** measure of the strength of the relationship.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}} $

<span class="code-span">r = 1</span>: Perfect positive linear relationship <br>
<span class="code-span">r = -1</span>: Perfect negative linear relationship <br>
<span class="code-span">r = 0</span>: No linear relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2022/07/Perfect-positive-correlation-Perfect-negative-correlation.webp">

</div>
</div>

<!--s-->

## Correlation | Significance

Almost all data can be determined to have a Pearson's correlation coefficient. To determine if the correlation is statistically significant, you can calculate the t-statistic and then find the p-value associated with it.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Getting a P Value

The t-statistic is calculated as:

$$ t = \frac{r \sqrt{n - 2}}{\sqrt{1 - r^2}} $$

Where:

- $r$ is the Pearson correlation coefficient.
- $n$ is the number of data points.

You null hypothesis is that there is no correlation between the two variables, i.e. $r = 0$.


</div>
<div class="c2" style = "width: 50%">

<div style='text-align: center;'>
   <img src='https://storage.googleapis.com/slide_assets/bad_pearsons.png' style='border-radius: 10px;'>
   <p style='font-size: 0.6em; color: grey;'>Nickolas 2024</p>
</div>

</div>
</div>


<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson's correlation coefficient is a great method to measure the strength of a linear relationship between two variables. However, it has some limitations:

- Sensitive to outliers
- It only measures linear relationships
- It is not robust to non-normality

If your data is not normally distributed, your relationship is not linear, or you have big outliers, you may want to consider another correlation method (e.g., Spearman's rank correlation coefficient).

<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2021/08/monotonic-relationships.png">

</div>
</div>

<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://datatab.net/assets/tutorial/spearman/Calculate_Spearman_rank_correlation.png">
<p style="text-align: center; font-size: 0.6em; color: grey;"> Source: Datatab</p>

</div>
</div>

<!--s-->

## OLAP | Correlation

Snowflake has built-in functions for calculating correlation coefficients. By default, it uses Pearson's correlation coefficient.

```sql

SELECT
    CORR(column1, column2) AS correlation_coefficient
FROM
    your_table;
```
<!--s-->

## L.06 | Q.01

You're working on a real estate prediction model. You want to know if there is a positive correlation between sq. feet in a house and it's sale price, specifically a linear relationship. Below is a plot of the relationship between the two variables, what correlation coefficient should you use to measure the strength of the relationship?

<div style = "max-height: 2vh;">
A. Pearson's <br>
B. Spearman's <br>
</div>

<div class = "col-wrapper" style = "align-items: top; justify-content: top;">
<div class="c1 col-centered" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1194/1*opIGrHAATX4NmdI6uzCSGA.png" style="border-radius: 10px; width: 100%; padding: 0px; margin: 0px;">
<p style="text-align: center; font-size: 0.6em; color: grey;"> Arshad (2024) </p>

</div>
<div class="c2 col-centered" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.01" width="100%" height="100%" style="border-radius: 10px"></iframe>

</div>
</div>

<!--s-->

## L.06 | Q.02

You're working in pharmaceuticals and want to know if there is a relationship between the dosage of a drug and binding affinity to a receptor. The relationship is non-linear, but you want to demonstrate that it is always increasing (i.e. higher dose == higher binding affinity). What correlation coefficient should you use to measure the strength of the relationship?

<div style = "max-height: 2.5vh;">
A. Pearson's <br>
B. Spearman's <br>
</div>

<div class = "col-wrapper" style = "align-items: top; justify-content: top;">
<div class="c1 col-centered" style = "width: 50%">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Monotonicity_example1.svg/1920px-Monotonicity_example1.svg.png" style="border-radius: 10px; width: 80%; padding: 5px; margin: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;"> Wikipedia (2025) </p>

</div>
<div class="c2 col-centered" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.02" width="100%" height="100%" style="border-radius: 10px"></iframe>

</div>
</div>

<!--s-->

<div class="header-slide">

# Simpson's Paradox

</div>

<!--s-->

## Simpson's Paradox

Simpson's Paradox is a phenomenon in probability and statistics, in which a trend appears in different groups of data but **disappears** or **reverses** when these groups are combined.

Recall that correlation measures the strength of a linear relationship between two variables. But, always remember that correlation does not imply causation! 

**Simpson's Paradox** is a situation where a relationship is reversed when the data is split into subgroups.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png">

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | 3% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in good condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | 1.3% |
| B | 6 | 594 | 1% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in poor condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | 4% |
| B | 57 | 1443 | 3.8% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Let's look at all of the data together. Hospital B has a higher death rate than Hospital A in aggregate. But, when we look at the subgroups, Hospital A has a higher death rate in both subgroups.

<div style="font-size: 0.8em;">

### Overall

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | **3%** |

### Good Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | **1.3%** |
| B | 6 | 594 | 1% |

### Poor Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | **4%** |
| B | 57 | 1443 | 3.8% |

</div>

<!--s-->

## Simpson's Paradox | Linear Regression Example

Simpson's Paradox can also occur in linear regression or correlation analysis.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png">

<!--s-->

## Simpson's Paradox | What Is (Typically) Happening?

1. **Confounding Variables**: The relationship between the variables is influenced by a third variable.
2. **Sample Size**: The sample size of the subgroups is not large enough to capture the true relationship.

<!--s-->

## Simpson's Paradox | Prevention

1. **Segment Data Carefully**: Understand the context and how data groups are formed.
2. **Identify Confounders**: Look for variables that might be influencing the results.
3. **Holistic Approach**: Consider both combined and segmented data analyses.
4. **Use Visualizations**: Visualizations can help identify patterns and trends.

<!--s-->

<div class="header-slide">

# Association Analysis

</div>

<!--s-->

## Association Analysis | Definition

Association analysis measures the strength of co-occurrence between one item and another. It is widely applied in retail analysis of transactions, recommendation engines, online clickstream analysis, and more.

<!--s-->

## Association Analysis | Explanation

Given a set of transactions, association analysis finds rules that will predict the occurrence of an item based on the occurrences of other items.

For example, if a customer buys a product, what other products are they likely to buy?

<!--s-->

## Association Analysis | Definitions

- **Itemset**: A collection of one or more items
  - Example: <span class="code-span">{Milk, Bread, Diaper}</span>
  - **k-itemset**: An itemset that contains <span class="code-span">k</span> items
- **Association Rule**: An implication expression of the form <span class="code-span">X --> Y</span>, where <span class="code-span">X</span> and <span class="code-span">Y</span> are itemsets.
  - Example: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

<!--s-->

## Association Rule | Evaluation

For a rule $ X \rightarrow Y $, where $ X \cap Y = \emptyset $, we can evaluate the rule using the following measures:

- **Support (S)**: Fraction of transactions that contain both X and Y. Where $\(T\)$ is the total number of transactions and $\sigma(X, Y)$ is the number of transactions that contain both $\(X\)$ and $\(Y\)$.

<div class="col-centered" style = "padding: 0.5em;">
$ S(X \rightarrow Y) = \frac{\sigma(X, Y)}{|T|} $
</div>

- **Confidence (C)**: Measures how often items in Y appear in transactions that contain X.

<div class="col-centered" style = "padding: 0.5em;">
$ C(X \rightarrow Y) = \frac{\text{S}(X, Y)}{\text{S}(X)} $
</div>

- **Lift (L)**: Takes into account the frequency of Y besides the confidence.

<div class="col-centered" style = "padding: 0.5em;">
$L(X \rightarrow Y) = \frac{S(X, Y)}{S(X)S(Y)}$
</div>

<!--s-->

## Association Rule | Example

Consider the following transactions:

| TID | Items |
| --- | --- |
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

And the following association rule: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

$ S = \frac{\sigma{\text(Milk, Diaper, Beer)}}{|T|} = \frac{2}{5} = 0.4 $

$ C = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)} = \frac{0.4}{0.6} = 0.67$

$ L = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)S(Beer)} = \frac{0.4}{0.6*0.6} = 1.11 $


<!--s-->

## Association Analysis Rule Generation

Given a set of transactions \(T\), the goal of association rule mining is to find all rules having:

- Support $ \geq $ Support threshold
- Confidence $ \geq $ Confidence threshold

The goal is to find all rules that satisfy these constraints. Lift is often used as a measure of the *interestingness* of the rule. Aka how much more likely is Y given X than if Y were independent of X.

<!--s-->

## Association Analysis Rule Generation | Brute-force Approach

In order to get all of the possible association rules, we would need to:

  - List all possible association rules
  - Compute the support and confidence for each rule
  - Prune rules that fail the support or confidence thresholds

But, as with many ideal or simple solutions, this is computationally prohibitive.

<!--s-->

## Association Analysis Rule Generation | Apriori Principle

The Apriori principle is a fundamental concept in association rule mining. It states that if an itemset is frequent, then all of its subsets must also be frequent. 

This principle allows us to reduce the number of itemsets we need to consider when generating association rules, and thus reduce the computational complexity. Modern software implementations will use the Apriori principle to generate association rules.

<div style='text-align: center;'>
  <img src='https://storage.googleapis.com/slide_assets/apriori.png' style='border-radius: 10px; margin: 0;'>
  <p style="text-align: center; font-size: 0.6em; color: grey; margin: 0;">Tank, Darshan. (2014)</p>
</div>


<!--s-->

## L.06 | Q.03

What is otherwise defined as the *interestingness* of an association rule?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Confidence <br>
&emsp;B. Support <br>
&emsp;C. Lift <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.06 | Q.03" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->


<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **EDA** concepts such as:

- Variance, Covariance, and Correlation
- Association Analysis

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->