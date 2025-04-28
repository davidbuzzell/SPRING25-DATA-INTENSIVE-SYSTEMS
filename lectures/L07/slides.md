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
  ## L.07 | EDA III

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
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 40%; padding-top: 5%; margin: 0;">
    <iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=Check In" width = "100%" height = "100%" style="margin: 0"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- Office hours reminders:
  - Thursdays (D'Arcy) from 2:00 PM to 3:00 PM in Mudd 3510, or by appointment.
  - Fridays (Mo) from 3:00 PM to 4:00 PM in Mudd First Floor Lobby. 

- Homework assignments
  - H.03 will be released by Monday (April 28) and due on Monday (May 5) at 11:59 PM.


<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **Hypothesis Testing** and **A/B Testing**?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->


<div class="header-slide">

# Hypothesis Testing

</div>

<!--s-->

## What is Hypothesis Testing?

Hypothesis testing uses statistical methods to determine whether there is enough evidence to reject a null hypothesis in favor of an alternative hypothesis.

The null hypothesis ($H_0$) is a statement that there is no difference between groups, while the alternative hypothesis ($H_1$) posits that there is a difference.

<!--s-->

## What is a P Value?

A p-value is the probability of observing a test statistic as extreme as the one calculated from the sample data, assuming that the null hypothesis is true.
A low p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, leading to its rejection.

p-values are widely used in industry and academia to demonstrate the significance of results. One of our primary goals with this lecture is to understand the p-value and its implications for decision making. 

<!--s-->

## L.07 | Q.01

A company is testing whether a new marketing strategy increases sales. The null hypothesis ($H_0$) is that the new strategy has no effect on sales. The p-value from the test is 0.03, and the significance level ($\alpha$) is set to 0.01. What should the company conclude?

<div class="col-wrapper">
<div class="c1" style="width: 50%; font-size: 0.9em">

A. Reject $H_0$ because the p-value is less than 0.05.  <br><br>
B. Fail to reject $H_0$ because the p-value is greater than $\alpha$.  <br><br>
C. Reject $H_0$ because the p-value is less than $\alpha$.  <br><br>
D. Fail to reject $H_0$ because the p-value is less than 0.05 but greater than $\alpha$.  <br><br>

</div>
<div class="c2" style="width: 50%;">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.01" width="100%" height="100%"></iframe>

</div>
</div>

<!--s-->

## Common Hypothesis Tests

The following are some common hypothesis tests used in statistics:

<div style="font-size: 0.7em;">

| Test | Assumptions | Usage (Easy ~ Rule) |
| --- | --- | --- |
| Independent t-test | 1. Data are independently and identically distributed. <br> 2. Both groups follow a normal distribution. <br> 3. Variances across groups are approximately equal.* | When comparing the means of two independent groups. |
| t-test (paired)  | 1. Data are independently and identically distributed. <br> 2. The differences are normally distributed.<br> 3. The pairs are selected randomly and are representative.| When you have pre / post test information on subjects or a matched pairs experiment. |
| chi-square test of independence | 1. Data are independently and identically distributed. <br> 2. All empirical frequencies are 5 or greater. | When comparing proportions across categories. |
| One-way ANOVA  | 1. Responses for each group are normally distributed. <br> 2. Variances across groups are approximately equal. <br> 3. Data are independently and identically distributed. | When comparing the means of three or more groups. |

</div>

<!--s-->

<div class="header-slide">

# Independent T-Test

</div>

<!--s-->

## Common Hypothesis Tests | Independent T-Test Setup

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

## Common Hypothesis Tests | Independent T-Test Calculation

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

## Common Hypothesis Tests | Independent T-Test Decision

### Decision Process

1. Compare the computed t-value against the critical t-value from the t-distribution table with $\alpha = 0.05$ and $df$.
2. If the computed t-value is higher than the critical t-value at $\alpha = 0.05$, reject the null hypothesis.

<div class="col-centered">
<img src="https://www.researchgate.net/publication/12025083/figure/fig1/AS:352960891637763@1461163842564/Extract-of-the-t-table-The-first-column-lists-the-degrees-of-freedom-n-1-The.png" style="border-radius: 10px; height: 50%; width: 40%;">
</div>

<!--s-->

## Common Hypothesis Tests | Independent T-Test Python

```python
import numpy as np
from scipy import stats

# Sample data
group1 = np.random.normal(8, 2, 40)
group2 = np.random.normal(7, 2, 50)

# Perform t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

<!--s-->

<div class="header-slide">

# Paired T-Test

</div>

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Setup
<div style="font-size: 0.9em">

### Scenario

A group of 25 patients is measured for cholesterol levels before and after a particular treatment, aiming to evaluate the treatment's effect on cholesterol.

### Hypotheses

- H0: $d=0$ (No difference in mean cholesterol levels)
- H1: $d \ne 0$ (Difference in mean cholesterol levels)

### Assumptions

- The differences within pairs are independent.
- The differences are normally distributed.
- The pairs are selected randomly and are representative.

</div>

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
<img src="https://www.researchgate.net/publication/12025083/figure/fig1/AS:352960891637763@1461163842564/Extract-of-the-t-table-The-first-column-lists-the-degrees-of-freedom-n-1-The.png" style="border-radius: 10px; height: 50%; width: 40%;">
</div>

<!--s-->

## T-Test (Paired) Python

```python
import numpy as np
from scipy import stats

# Sample data
before = np.array([200, 210, 220, 230, 240])
after = np.array([190, 200, 210, 220, 230])

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

<!--s-->

## L.07 | Q.02

Let's say a professor wants to compare the exit polls vs the intro polls for individual students in his class to assess the efficacy of his lectures. What is the null hypothesis? What test should he use?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

A. $H_0$ is that the exit poll and intro poll are the same. Paired t-test. <br><br>
B. $H_0$ is that the exit poll and intro poll are different. Indepenent t-test. <br><br>
C. $H_0$ is that the exit poll and intro poll are the same. Independent t-test. <br><br>

</div>
<div class="c2" style = "width: 50%">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.02" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

<div class="header-slide">

# Chi-Square Test

</div>

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Setup

### Scenario

You have two stores, Store A and Store B. The following data is collected:

| Sale | Store A | Store B |
|---------|----------|----------|
| Football  | 15       | 5       |
| Soccer Ball | 5     | 15       |


### Hypotheses

- H0: The Sale Item distribution is independent of the Store.
- H1: The Sale Item distribution is dependent on the Store.

### Assumptions

- Observations are independent.
- All expected frequencies are at least 5.

<!--s-->

## L.07 | Q.03

**Question:** Calculate the expected frequency for footballs in Store A. Assuming the null hypothesis is true, what is the expected frequency?

| Sales | Store A | Store B |
|---------|----------|----------|
| Football  | 15       | 5       |
| Soccer Ball | 5     | 15       |


<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: 3em; padding-left: 0em;'>
<div class='c1' style = 'width: 40%; display: flex; align-items: left; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
A. 10<br>
B. 5<br>
C. 7.5<br>
D. 12.5<br>
</div>
</div>
<div class='c2' style = 'width: 60%; display: flex; align-items: center; flex-direction: column;'>

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.03" width = "100%" height = "100%"></iframe>

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

<div class="header-slide">

# One-Way ANOVA

</div>

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Setup

### Scenario

An MBA program is evaluating the effectiveness of three teaching methods: traditional lectures, case-based learning, and experiential learning. Students are randomly assigned to one of the three methods, and their final exam scores are recorded. The goal is to determine if there is a significant difference in the mean exam scores across the three teaching methods.

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
| Kruskal–Wallis H  | ANOVA | For 3 or more groups of quantitative data. |
| Mann-Whitney U | Independent t-test  | For 2 groups, different populations. |
| Wilcoxon Signed-rank  | Paired t-test| For 2 groups from the same population. |

<p style = "text-align: center; color: grey"> © Adapted from Scribbr, 2024 </p>

</div>

<!--s-->

<div class="header-slide">

# A/B Testing

</div>

<!--s-->

## Introduction to A/B Testing

A/B testing, a critical methodology widely used in tech companies and various industries, plays a pivotal role in optimizing user experience and boosting key performance indicators such as engagement and conversion rates.

This approach tests two variants, A and B, in a randomized experiment to make data-driven decisions regarding changes to products, websites, or apps.

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Posta (2024)</p>

<!--s-->

## Understanding A/B Testing

At its core, A/B testing involves comparing two versions of a web page, product feature, or other elements to determine which one performs better.

This process enables businesses to make incremental changes that can lead to significant improvements in user satisfaction and business outcomes.

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Posta (2024)</p>

<!--s-->

## Why Implement A/B Testing?

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Innovation and Improvement
Whether it's deciding which photograph leads to better retention of subscribers or testing a new feature's impact on user engagement, A/B testing provides a systematic method for assessing the potential benefits of new ideas.

</div>
<div class="c2" style = "width: 50%">

### Data-driven Decisions
By relying on empirical data, companies can move beyond guesswork and make informed decisions that align with their strategic goals.

</div>
</div>

<!--s-->

## Case Study | Social Network App

A study within a new social network app revealed that users who tagged a friend during their trial period had a **31**% increase in daily usage after the trial period ended.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

This insight suggests that encouraging social interactions may **increase user retention**.

An immediate reaction might be to prioritize features solely based on this correlation, such as pushing all new installs to tag a friend.

</div>
<div class="c2" style = "width: 50%">

<iframe width = "100%" height = "60%" src="https://lottie.host/embed/6a06ca79-cb9c-48e2-bdac-a8bbbcb03815/j0VNBrf81N.json"></iframe>

</div>
</div>

<!--s-->

## L.07 | Q.04

After observing the correlation between tagging a friend and increased user retention, should the app developers immediately implement this feature for all new users?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 60%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Yes <br>
&emsp;B. No <br>
</div>
</div>

<div class="c2 col-centered" style = "width: 40%;">
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L.07 | Q.04" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Case Study | Considering External Influences

Factors such as:

- **Seasonality:** Changes in user behavior due to holidays or seasonal trends.
- **Marketing Campaigns:** The impact of marketing campaigns on user engagement.
- **Product Updates:** Changes in the app's features or design.
- **User Demographics:** Variations in user behavior based on age, location, or other factors.

Could all contribute to conversion rate changes. Thus, it's imperative not to jump to conclusions about causality based solely on passively observed correlations.

<!--s-->

## Case Study | Implementing Effective A/B Testing

<div style="font-size: 0.9em">
To understand the impact of tagging a friend during a trial period, a structured A/B testing approach is essential:

1. **Construct a Hypothesis:** Begin with a clear, testable hypothesis.
    - **H0**: Tagging a friend during the trial period has no effect on the original users' retention.
    - **H1**: Tagging a friend during the trial period increases the original users' retention.

2. **Design Study / Trial:** Establish how many subjects are necessary for a statistically significant test and estimate the duration of the experiment. 
    - Resources exist to make this process easier [[link]](https://www.evanmiller.org/ab-testing/sample-size.html)

3. **Measure Results:** Use hypothesis testing to assess if there's a significant difference between the two groups.
    - We'll cover these tests in more detail later in the lecture.

4. **Take Action:** Based on the findings, make informed decisions and continue the cycle of experimentation and learning.
</div>

<!--s-->

## Proper A/B Testing Highlights

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Randomization
Randomly assign subjects to the control and treatment groups to avoid selection bias.

### Statistical Significance
Use statistical tests to determine if the observed differences are significant.

### Sample Size
Ensure that the sample size is large enough to detect meaningful differences.

</div>
<div class="c2" style = "width: 50%">

### Duration
Run the experiment for a sufficient duration to capture the effects of the changes.

### Segmentation
Analyze results across different segments to identify potential variations in the treatment effect.

</div>
</div>


<!--s-->


<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with **Hypothesis Testing** and **A/B Testing**?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Additional Slides
## (for the curious)

</div>

<!--s-->

## Wondering what a T Distribution Looks Like?

Here's how you can visualize a t-distribution in Python (and better understand T-tables!):

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

```python
import numpy as np
import plotly.express as px
from scipy.stats import t

x = np.linspace(-5, 5, 1000)
y = t.pdf(x, df=4)
fig = px.line(x=x, y=y)
fig.update_layout(title='t-distribution with 4 degrees of freedom', template = "plotly_white")
```

<img src="https://storage.googleapis.com/slide_assets/p-value.png">

</div>
<div class="c2" style = "width: 30%; font-size: 0.6em;">

PDF of the t-distribution can be calculated as:

$$ f(x, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}} $$

Where $\nu$ is the degrees of freedom.

</div>
</div>

<!--s-->

## Why df = n - 1?

We often subtract 1 from the sample size to account for the fact that we are estimating the population mean from the sample mean. But *why* do we do this?

To illustrate:

- Recall that the definition of degrees of freedom is the number of **independent** observations in a sample.
- Suppose you have a sample of 5 data points: <span class="code-span"> [1, 2, 3, 4, 5] </span>
- You calculate the sample mean: <span class="code-span"> 3 </span>
- You can calculate the last data point from the sample mean and 4 of the other data points. So you lose a degree of freedom, because you can't change the last data point without changing the sample mean.

<!--s-->

## Common Hypothesis Tests | T-Test Calculation

<div style="font-size: 0.8em">

### Calculating the T-Statistic (Not Equal Variances, Welch's T-Test)
  
`$$ t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$`
  
Where:  
  - $\bar{x}_1$ and $\bar{x}_2$ are the sample means.
  - $n_1$ and $n_2$ are the sample sizes.
  - $s_1$ and $s_2$ are the sample standard deviations. <br>

### Calculating the Degrees of Freedom (Not Equal Variances, Welch's T-Test)

$$ df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1} + \frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2 - 1}} $$

Where:
  - $s_1$ and $s_2$ are the sample standard deviations.
  - $n_1$ and $n_2$ are the sample sizes.

</div>