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
  ## L.09 | Dimensionality Reduction & Exam Review

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

- H.03 will be due on Monday, 05.05.2025 @ 11:59PM. 
  - A couple of errors in the tests noted.
- Office hours tomorrow (Friday) from 3:00PM - 4:00PM.
- Exam Part I will be on Monday, 05.05.2025 @ 3:30PM - 4:50PM.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you feeling about Exam Part I?


  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Dismensionality Reduction

</div>

<!--s-->

## High-Dimensional Data | Why Reduce Dimensions?

- **Curse of Dimensionality**: As the number of features increases, the amount of data required to cover the feature space grows exponentially.

- **Overfitting**: High-dimensional data is more likely to overfit the model, leading to poor generalization.

- **Computational Complexity**: High-dimensional data requires more computational resources to process.

- **Interpretability**: High-dimensional data can be difficult to interpret and visualize.

<!--s-->

## High-Dimensional Data | The Curse of Dimensionality

**tldr;** As the number of features increases, the amount of data required to cover the feature space grows exponentially. This can lead to overfitting and poor generalization.

**Intuition**: Consider a 2D space with a unit square. If we divide the square into 10 equal parts along each axis, we get 100 smaller squares. If we divide it into 100 equal parts along each axis, we get 10,000 smaller squares. The number of smaller squares grows exponentially with the number of divisions. Without exponentially growing data points for these smaller squares, a model needs to make more and more inferences about the data.

**Takeaway**: With regards to machine learning, this means that as the number of features increases, the amount of data required to cover the feature space grows exponentially. Given that we need more data to cover the feature space effectively, and we rarely do, this can lead to overfitting and poor generalization.

<img src="https://storage.googleapis.com/slide_assets/dimensionality.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Rajput, 2012</p>

<!--s-->

## Dimensionality Reduction | Common Techniques

### Covered in L.09

- **Feature Selection**: Selecting a subset of the most important features.
- **Feature Extraction**: Creating new features by combining existing features.

### Covering Today

- **PCA**: A technique for reducing the dimensionality of data by projecting it onto a lower-dimensional subspace.
- **t-SNE**: A technique for visualizing high-dimensional data by reducing it to 2 or 3 dimensions.
- **Autoencoders**: Neural networks that learn to compress and reconstruct data.

<!--s-->

## High-Dimensional Data | Principal Component Analysis (PCA)

PCA reduces dimensions while preserving data variability. PCA works by finding the principal components of the data, which are the directions in which the data varies the most. It then projects the data onto these principal components, reducing the dimensionality of the data while preserving as much of the variability as possible.

<img src = "https://numxl.com/wp-content/uploads/principal-component-analysis-pca-featured.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NumXL</p>

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(data)
pca.transform(data)
```

<!--s-->

## High-Dimensional Data | Principal Component Analysis (PCA) Example

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python
import numpy as np

def pca_with_numpy(data, n_components=None):
    """Calculate PCA using numpy."""

    # Center data.
    centered_data = data - np.mean(data, axis=0) 

    # Calculate covariance matrix.
    cov_matrix = np.cov(centered_data.T)

    # Eigenvalue decomposition.
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues in descending order.
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top components.
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    # Project data onto principal components.
    transformed_data = np.dot(centered_data, eigenvectors)
    return transformed_data, eigenvectors, eigenvalues

```

</div>
<div class="c2" style = "width: 50%">

```python
from sklearn.decomposition import PCA

def pca_with_sklearn(data, n_components=None):
    """Calculate PCA using sklearn."""
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca.components_, pca.explained_variance_
```

</div>
</div>

<!--s-->

## High-Dimensional Data | T-distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a technique for visualizing high-dimensional data by reducing it to 2 or 3 dimensions. 

t-SNE works by minimizing the divergence between two probability distributions: one that describes the pairwise similarities of the data points in the high-dimensional space and another that describes the pairwise similarities of the data points in the low-dimensional space.

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne.fit_transform(data)
```

<!--s-->

## High-Dimensional Data Example

Here you can see the Iris dataset visualized in 3D. The dataset contains 3 features: sepal length, sepal width, and petal length. The data points are colored by their species. Our goal is to represent the data in 2D while preserving the relationships between the data points.

<iframe width= "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/iris_3d_scatter_plot.html" title="scatter_plot"></iframe>

<!--s-->

## High-Dimensional Data Example | PCA vs t-SNE

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-bottom: -20em;">

<iframe width = "100%" height = "80%" src="https://storage.googleapis.com/slide_assets/iris_pca_2d_scatter_plot.html" title="scatter_plot" ></iframe>

</div>
<div class="c2" style = "width: 50%; margin-bottom: -20em;">

<iframe width = "100%" height = "80%" src="https://storage.googleapis.com/slide_assets/iris_tsne_2d_scatter_plot.html" title="scatter_plot" ></iframe>

</div>
</div>

<!--s-->

## High-Dimensional Data | Autoencoders

Autoencoders are neural networks that learn to compress and reconstruct data. They consist of an encoder that compresses the data into a lower-dimensional representation and a decoder that reconstructs the data from this representation.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; ">

```python

from keras.layers import Input, Dense
from keras.models import Model

input_data = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='sigmoid')(input_data)
decoded = Dense(input_dim, activation='sigmoid')(encoded)
autoencoder = Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(data, data, epochs=50, batch_size=256, shuffle=True)
```

</div>
<div class="c2" style = "width: 50%">

<img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Autoencoder_schema.png/500px-Autoencoder_schema.png" width="100%" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Wikipedia 2019</p>

</div>
</div>

<!--s-->

<div class="header-slide">

# Exam Part I Review

</div>

<!--s-->

## Exam Part I Overview

- **Format**: 50% multiple choice (\~20 questions, 1 pt each), 50% Free Response (\~10 questions, 2 pts each), closed book format.
  - **Multiple choice**: Classic format, select the best answer(s).
  - **Free Response**: Infrastructure choices, exploring data, interpreting results, DIS decision-making.  
- **Duration**: Monday, 05.05.2025, 3:30 PM - 4:50 PM.
- **Content**: All material covered in class, including lectures, readings, and assignments are fair game. This review will cover the majority of the content.

<!--s-->

## L.02 | Databases

### Topics to Review

- What makes a good database?
- Basic SQL
- Database landscape

<!--s-->

## L.02 | Databases

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

You're tasked with selecting a database for a new project in the e-commerce domain. The project will handle a large volume of transactions and requires high availability. Which database type would be the most suitable for this project?

&emsp;A. Relational Database <br><br>
&emsp;B. NoSQL Database <br><br>
&emsp;C. Graph Database <br><br>
&emsp;D. Columnar Database

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You work at a high-frequency trading firm and require a database that has extremely low latency and high throughput for read and write operations. Which database type would you choose and why? Discuss the trade-offs involved in your choice compared to an alternative.

</div>
</div>

<!--s-->

## L.03 | Databases II

### Topics to Review

- Connecting to databases
- Essential security concepts

<!--s-->

## L.03 | Databases II

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

Client side applications are designed to run on the client's computer, while server side applications are designed to run on the server. Let's say you have an application running in a client's browser. How would you connect to a database from this application?

&emsp;A. A. Use a database driver like psycopg2 to connect to the database.<br><br>
&emsp;B. Use a REST API to connect to the database.<br><br>
&emsp;C. Use a CLI tool like psql to connect to the database.<br><br>

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You are tasked with securing a database system for a financial application. Describe the principle of least privilege and explain how you would apply it to secure database access. Use the following terms in your answer: "user roles" and "permissions". 

</div>
</div>

<!--s-->

## L.04 | Online Analytical Processing (OLAP)

### Topics to Review

- OLTP vs OLAP
- Normalized vs Denormalized Schemas

<!--s-->

## L.04 | Online Analytical Processing (OLAP)

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

What is **normalized** data in the context of OLTP systems?

&emsp;A. Data that is stored in a single table with no relationships.<br><br>
&emsp;B. Data that is stored in multiple tables with relationships between them.<br><br>

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You are a technical product manager at a rapidly scaling startup in the textiles industry. You're given an extremely large dataset (~5TB) to analyze. It contains data on the ecological impact of your business across the globe. Describe how you would approach this task using OLAP systems. Ensure you use the terms "OLTP", "OLAP", and "modeling" in your answer.

</div>
</div>

<!--s-->

## L.05 | OLAP + EDA I

### Topics to Review

- Data imputation
- Outlier detection

<!--s-->

## L.05 | OLAP + EDA I


<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

You are working for a cybersecurity company tasked with detecting anomalies in network traffic data. The dataset contains hundreds of features, such as packet size, source and destination IPs, protocols, and timestamps. The data is high-dimensional and not normally distributed.

Select the most appropriate method for detecting anomalies in this dataset.

&emsp;A. IQR<br><br>
&emsp;B. Z-Score<br><br>
&emsp;C. Isolation Forest<br><br>

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You identify missing values in your tabular dataset. You have two options: drop the rows with missing values or impute them. Discuss the pros and cons of each approach. Then, describe an imputation method you may use and why you chose it.

</div>
</div>

<!--s-->

## L.06 | OLAP + EDA II

### Topics to Review

- Variance, Covariance, and Correlation
- Association Rules

<!--s-->

## L.06 | OLAP + EDA II

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

You are working with a dataset containing information about customer purchases in an online store. You want to find patterns in the data to understand which products are frequently bought together. Which of the following techniques would be most appropriate for this task?

&emsp;A. Association Analysis <br><br>
&emsp;B. Pearson Correlation <br><br>
&emsp;C. Spearman Correlation

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You're working in the e-commerce industry where the machine learning task is to correlate the number of reviews a product has with its average rating. You do not care if there is a linear relationship or not, and in fact, you suspect there isn't. What correlation coefficient would you use to measure the correlation between the two variables? In other words, how can you measure the correlation between the number of reviews and the average rating always increasing together? Include the following terms in your answer: "correlation", "Pearson", and "Spearman".

</div>
</div>


<!--s-->

## L.07 | OLAP + EDA III

### Topics to Review

- Hypothesis Testing
- A/B Testing

<!--s-->

## L.07 | OLAP + EDA III

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

You work for a sports analytics company and are tasked with analyzing the performance of two different training programs for athletes. You want to determine if there is a statistically significant difference in performance between the two programs. Your outcome variable is continuous (e.g. winning spread in a game) and normally distributed. Which statistical test would you use?

&emsp;A. Chi-Squared Test <br><br>
&emsp;B. T-Test <br><br>
&emsp;C. ANOVA <br><br>
&emsp;D. Mann-Whitney U Test

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You are working for a large e-commerce company and are tasked with analyzing the impact of a new website design on conversion rates. Describe a process to assess the impact of the new design. Include the following terms in your answer: "A/B testing", "hypothesis testing", "p-value", and "statistical significance".

</div>
</div>


<!--s-->

## L.08 | OLAP + EDA IV

### Topics to Review

- Regular Expressions
- Semantic Search & Embeddings
- Visualizing Embeddings

<!--s-->

## L.08 | OLAP + EDA IV

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Example Multiple Choice

You are working for a company and are tasked with analyzing customer reviews. You want to extract all email addresses from the reviews. Which of the following regular expressions would be *most* appropriate for this task?

&emsp;A. <span class = "code-span">[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}</span> <br><br>
&emsp;B. <span class = "code-span">[0-9]{3}-[0-9]{2}-[0-9]{4}</span> <br><br>  
&emsp;C. <span class = "code-span">https?://[^\s]+</span> <br><br>  
&emsp;D. <span class = "code-span">[A-Z]{5}[0-9]{4}[A-Z]{1}</span> <br><br>  

</div>
<div class="c2" style = "width: 50%">

### Example Free Response

You are working for a company and are tasked with analyzing customer reviews. You want to find all positive reviews so that you can use them in a marketing compaign. Describe a process to extract the positive reviews. Ensure you define and include the following terms in your answer: "semantic search" and "embeddings".

</div>
</div>

<!--s-->

## L.09 | Distributed Preprocessing I

### Topics to Review

- What is an ML-Ready Dataset
- Transformations of data (Numerical, Categorical, Text)
- Feature Selection

<!--s-->

## L.09 | Distributed Preprocessing I

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

<div style='text-align: center;'>
   <img src='https://discovery.cs.illinois.edu/static/learn/NC-WebG.png' style='border-radius: 10px; width: 50%; margin-right: 2em;'>
</div>

Look at the above histogram of the distribution of a feature. What is the best way to transform this feature before using it in a machine learning model?

&emsp;A. Standardization <br><br>
&emsp;B. Min-Max Scaling <br><br>
&emsp;C. Log Transformation

</div>

<div class="c2" style = "width: 50%">

### Example Free Response

You're given a dataset with 900 features, and you suspect that many of them are irrelevant or redundant. Describe a process to select the most important features for your machine learning model. Ensure you define and include the following terms in your answer: "feature selection".

</div>

<!--s-->

## L.10 | Distributed Preprocessing II

### Topics to Review

- Curse of Dimensionality
- Dimensionality Reduction (PCA, Autoencoders)

<!--s-->

## L.10 | Distributed Preprocessing II

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Example Multiple Choice

Let's say you need to compress a dataset with 100 features into 20 features. Interpretability of the remaining features is critically important. Which of the following methods should you use?

&emsp;A. PCA <br><br>
&emsp;B. Autoencoders <br><br>
&emsp;C. Feature Selection w/ Chi-Squared


</div>
<div class="c2" style = "width: 50%">

### Example Free Response

Describe a process to reduce the dimensionality of a dataset with 1000 features. For your task, you need a fast and easily reproducible method for dimensionality reduction, and you don't care very much about the interpretability of your resulting features. What method would you use? Ensure you define and include the following terms in your answer: "curse of dimensionality", "PCA", and "autoencoders".

</div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you feeling about Exam Part I?


  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->