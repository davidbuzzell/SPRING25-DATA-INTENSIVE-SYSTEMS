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
  ## L.05 | OLAP + EDA II

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

# OLAP + EDA II

</div>

<!--s-->

## Agenda

### Variance, Covariance, and Correlation

**Scenario**: You are a data analyst at a large e-commerce company. You have been tasked with analyzing the relationship between customer spending and the number of items purchased. You have access to a dataset that contains information on customer spending and the number of items purchased for each transaction.

### Hypothesis Testing

**Scenario**: You are a data analyst at a large e-commerce company. You have been tasked with analyzing the effectiveness of a new marketing campaign. You have access to a dataset that contains information on customer spending before and after the campaign was launched, which is a form of A/B testing.

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

## Variance

Variance is the average of the squared differences from the mean for a single variable.

$$ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$

<!--s-->

## Correlation | Quantitative Measurement via Covariance

**Covariance** is a measure of how much two random variables vary together, which is a measure of their **correlation**.

The covariance between two variables \(X\) and \(Y\) can be defined as:

<div class="col-centered">
$ \text{cov}(X, Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n} $
</div>

And it's really just a generalization of the variance to two variables:

<div class="col-centered">

$ \sigma^2 = \frac{\sum_{i=1}^n (X_i - \mu)^2}{n} $

</div>

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

Pearson correlation coefficient, denoted by \(r\), is a measure of the linear correlation between two variables. It ranges from -1 to 1, and so it is a **standardized** measure of the strength of the relationship.

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

<div class="header-slide">

# Hypothesis Testing

</div>

<!--s-->

## Common Hypothesis Tests

<div style="font-size: 0.7em;">

| Test | Assumptions | Usage (Easy ~ Rule) |
| --- | --- | --- |
| t-test | 1. Data are independently and identically distributed. <br> 2. Both groups follow a normal distribution. <br> 3. Variances across groups are approximately equal.* | When comparing the means of two independent groups. |
| t-test (paired)  | 1. Data are independently and identically distributed. <br> 2. The differences are normally distributed.<br> 3. The pairs are selected randomly and are representative.| When you have pre / post test information on subjects or a matched pairs experiment. |
| chi-square test of independence | 1. Data are independently and identically distributed. <br> 2. All empirical frequencies are 5 or greater. | When comparing proportions across categories. |
| One-way ANOVA  | 1. Responses for each group are normally distributed. <br> 2. Variances across groups are approximately equal. <br> 3. Data are independently and identically distributed. | When comparing the means of three or more groups. |

</div>

<!--s-->

## Common Hypothesis Tests | T-Test Setup

<div style="font-size: 0.9em">

### Scenario

Comparing the effect of two medications. Medication A has been used on 40 subjects, having an average recovery time of 8 days, with a standard deviation of 2 days. Medication B (new) has been used on 50 subjects, with an average recovery time of 7 days and a standard deviation of 2.5 days. 

### Hypotheses

- H0: μ1 = μ2 (No difference in mean recovery time)
- H1: μ1 ≠ μ2 (Difference in mean recovery time)

### Assumptions

- Groups are I.I.D.
    - I.I.D. stands for independent and identically distributed.
- Both groups follow a normal distribution.*
    - Once you have enough samples, the central limit theorem will ensure normality.
- Equal variances between the two groups (homoscedasticity).*
    - If variances are not equal, a Welch's t-test can be used.

</div>

<!--s-->

## Common Hypothesis Tests | T-Test Calculation

<div style="font-size: 0.9em">

### T-Statistic (Equal Variances)

`$$ t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} $$`

`$$ s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}} $$`

Where:

- $\bar{x}_1$ and $\bar{x}_2$ are the sample means.
- $n_1$ and $n_2$ are the sample sizes.
- $s_p$ is the pooled standard deviation.
- $s_1$ and $s_2$ are the sample standard deviations. <br>

### Degrees of Freedom (Equal Variances)

The degrees of freedom for this t-test is calculated as: 

$$ df = n_1 + n_2 - 2 $$

</div>

<!--s-->

## Common Hypothesis Tests | T-Test Decision

### Decision Process

1. Compare the computed t-value against the critical t-value from the t-distribution table with $\alpha = 0.05$ and $df$.
2. If the computed t-value is above the critical t-value, reject the null hypothesis.

<div class="col-centered">
<img src="https://www.researchgate.net/publication/12025083/figure/fig1/AS:352960891637763@1461163842564/Extract-of-the-t-table-The-first-column-lists-the-degrees-of-freedom-n-1-The.png" style="border-radius: 10px; height: 50%; width: 50%;">
</div>

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Setup

### Scenario

A group of 25 patients is measured for cholesterol levels before and after a particular treatment, aiming to evaluate the treatment's effect on cholesterol.

### Hypotheses

- H0: $d=0$ (No difference in mean cholesterol levels)
- H1: $d \ne 0$ (Difference in mean cholesterol levels)

### Assumptions

- The differences within pairs are independent.
- The differences are normally distributed.
- The pairs are selected randomly and are representative.

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Calculation

### Paired T-Statistic

First, find the difference ($d$) for each pair. Then, calculate the mean ($\bar{d}$) and standard deviation ($s_d$) of those differences.

$$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$

where $n$ is the number of pairs.

### Degrees of Freedom

Degrees of freedom can be calculated with $df = n - 1$.

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Decision

### Decision Process

1. Using the t-distribution table with $df = n - 1$, compare the calculated t-value.
2. If the computed t-value falls within the critical range, reject the null hypothesis.

<div class="col-centered">
<img src="https://www.researchgate.net/publication/12025083/figure/fig1/AS:352960891637763@1461163842564/Extract-of-the-t-table-The-first-column-lists-the-degrees-of-freedom-n-1-The.png" style="border-radius: 10px; height: 50%; width: 50%;">
</div>


<!--s-->

## Common Hypothesis Tests | Chi-Square Test Setup

### Scenario

You have two penguin species, Adelie and Chinstrap. They are observed in the wild, and the following data is collected from two islands (A and B):

| Species | Island A | Island B |
|---------|----------|----------|
| Adelie  | 15       | 5       |
| Chinstrap | 5     | 15       |


### Hypotheses

- H0: The species distribution is independent of the island.
- H1: The species distribution is dependent on the island.

### Assumptions

- Observations are independent.
- All expected frequencies are at least 5.

<!--s-->

## Q.01 | Chi-Square Expectation Calculation

**Question:** Calculate the expected frequency for Adelie penguins on Island A. Assuming the null hypothesis is true, what is the expected frequency?

<div class='col-wrapper' style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

| Species | Island A | Island B |
|---------|----------|----------|
| Adelie  | 15       | 5       |
| Chinstrap | 5     | 15       |

&emsp;A. 10<br>
&emsp;B. 5<br>
&emsp;C. 7.5<br>
&emsp;D. 12.5<br>
</div>
<div class='c2 col-centered' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com?label=L.06|Q.01" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Calculation

### Chi-Square Statistic

The chi-square statistic of independence is calculated as:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where:
- $O$ is the observed frequency.
- $E$ is the expected frequency, which is calculated as the row total times the column total divided by the grand total.

### Degrees of Freedom

$$df = (r - 1) \times (c - 1)$$

Where:
- $r$ is the number of rows.
- $c$ is the number of columns.

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Calculation

### Calculation ($\chi^2$)

$$ \chi^2 = \frac{(15 - 10)^2}{10} + \frac{(5 - 10)^2}{10} + \frac{(5 - 10)^2}{10} + \frac{(15 - 10)^2}{10} = 10 $$

### Degrees of Freedom ($df$)

$$ df = (2 - 1) \times (2 - 1) = 1 $$

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Decision

### Decision Process

1. Compare the $\chi^2$ value against the critical values from the chi-square distribution table with $df$
2. If $\chi^2 > \chi_{critical}$, reject H0.

<div class="col-centered">
<img src="https://www.mun.ca/biology/scarr/IntroPopGen-Table-D-01-smc.jpg" style = "border-radius: 10px; height: 40%; width: 40%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">© 2022, Steven M. Carr</p>

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Setup

### Scenario

Testing if there's a difference in the mean test scores across three teaching methods used across different groups.

### Hypotheses


- H0: $ \mu_1 = \mu_2 = \mu_3 $ (no difference among the group means)
- H1: At least one group mean is different.

### Assumptions

- Groups are I.I.D.
- Groups follow a normal distribution.
- Variances across groups are approximately equal.
    - A good rule of thumb is a ratio of the largest to the smallest variance less than 4.

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Calculation

### F-Statistic

Anova breaks down the variance explained by the groups ($SS_{between}$) and the variance not explained by the groups ($SS_{within}$). The F-statistic measures the ratio of the variance between groups to the variance within groups:

$$ F = \frac{SS_{between} / df_{between}}{SS_{within} / df_{within}} $$

The total sum of squares (SS) is calculated as:

$$ s^2 = \frac{SS}{df} = \frac{\sum (x - \bar{x})^2}{n - 1} $$

Where:
- $SS$ is the sum of squares.
- $df$ is the degrees of freedom.
- $x$ is the data point.
- $\bar{x}$ is the sample mean.
- $n$ is the sample size.

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Calculation

### Degrees of Freedom

The degrees of freedom are $df_{between} = k - 1$ and $df_{within} = N - k$.

Where:
- $k$ is the number of groups.
- $N$ is the total number of observations.


<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Decision

### Decision Process

1. Compare the calculated F-value with the critical F-value from the F-distribution table at $df_{between}$ and $df_{within}$.
2. Reject H0 if $F > F_{critical}$, indicating significant differences among means.

<div class="col-centered">
<img src="https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha05.png?resize=817%2C744&ssl=1" height = "50%" width = "50%" style="border-radius: 10px;">
</div>

<!--s-->

## Choosing a Non-Parametric Test

If the assumptions for parametric tests are not met, non-parametric tests can be used. 

These tests are distribution-free and do not require the data to be normally distributed. These may make less powerful inferences than parametric tests, because parametric tests derive power from the strong assumptions they make about the shape of the data.

<div style="font-size: 0.8em">

| Test    | Use in place of | Description |
|-----------------------|------------------|-------------------------|
| Spearman’s r  | Pearson’s r | For quantitative variables with non-linear relation. |
| Kruskal–Wallis H  | ANOVA | For 3 or more groups of quantitative data |
| Mann-Whitney U | Independent t-test  | For 2 groups, different populations. |
| Wilcoxon Signed-rank  | Paired t-test| For 2 groups from the same population. |

<p style = "text-align: center; color: grey"> © Adapted from Scribbr, 2024 </p>

</div>

<!--s-->

## OLAP | Snowflake

Snowflake does not have built-in functions for hypothesis testing. But, you can use SnowPark. This example uses remote resources for the filtering, and local resources for the calculation.

```python

import snowflake.snowpark as sp
from snowflake.snowpark import functions as F
from scipy import stats

def t_test(df, col1, col2):
    group1 = df.filter(F.col('group') == 'A').select(col1).toPandas()[col1]
    group2 = df.filter(F.col('group') == 'B').select(col2).toPandas()[col2]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return t_stat, p_value

df = session.table('your_table')
result = t_test(df, 'column1', 'column2')
```


