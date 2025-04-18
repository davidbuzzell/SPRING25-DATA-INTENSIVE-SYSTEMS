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

### Covered in L.08

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

## High-Dimensional Data | T-distributed Stochastic Neighbor Embedding (t-SNE)

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/iris_3d_plot.html" title="scatter_plot" padding=2em;></iframe>


<!--s-->

## High-Dimensional Data | T-distributed Stochastic Neighbor Embedding (t-SNE)

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/slide_assets/iris_2d_plot.html" title="scatter_plot" padding=2em;></iframe>

<!--s-->

## High-Dimensional Data | Autoencoders

Autoencoders are neural networks that learn to compress and reconstruct data. They consist of an encoder that compresses the data into a lower-dimensional representation and a decoder that reconstructs the data from this representation.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

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

# Exam Review

</div>

<!--s-->



