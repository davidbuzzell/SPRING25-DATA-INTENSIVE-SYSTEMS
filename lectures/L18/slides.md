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
  ## L.18 | MLOps and Exam Review

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
    <iframe src = "https://drc-cs-9a3f6.web.app?label=Check In" width = "100%" height = "100%"></iframe>
  </div>
</div>

<!--s-->

## Announcements

- **Final Exam**: The final exam will be held on **June 5th**. It will cover all topics discussed in the course.

- **Office Hours**: No office hours on June 5th. Please reach out if you want to meet earlier on June 5th! 

- **CTEC**: We use CTEC to collect feedback, and these are now open. Your feedback is important to me! Please fill out the [CTEC](https://canvas.northwestern.edu/courses/231041/external_tools/8871) here.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with multiprocessing, data & model parallelism, and monitoring platforms?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# MLOps Continued

</div>

<!--s-->

## DIS (~ 5 minutes)

Most industry teams are not researching new machine learning models. Instead, they leverage existing models and focus on optimizing the data and pipelines that feed into those models. The key differentiator is often how effectively you use available tools and how quickly you can iterate.

**With that in mind, what strategies can you use to speed up your ML pipeline?**

<!--s-->

<div class="header-slide">

# Local Python Multiprocessing

</div>

<!--s-->

## Local Python Multiprocessing
Fast local data preprocessing is essential for efficient machine learning workflows. It allows for quick data preparation, enabling faster iterations and experimentation. One of the key tools for achieving this in Python is the built-in <span class="code-span">multiprocessing</span> module.

### Benefits of Fast Local Data Preprocessing

- **Reduced Overall Iteration Time**: Efficient preprocessing reduces the overall time required for the data & model lifecycle.
- **Increased Productivity**: Faster data preparation allows for more iterations and experiments within the same timeframe.
- **Resource Optimization**: Utilizes local CPU resources effectively, reducing the need for expensive cloud-based solutions.

<!--s-->

## Python's <span class="code-span">multiprocessing</span> Module

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

The <span class="code-span">multiprocessing</span> module in Python provides a simple and powerful way to parallelize data preprocessing tasks across multiple CPU cores. This can significantly speed up the preprocessing pipeline.

</div>
<div class="c2" style = "width: 30%; text-align: center;">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/640px-Python-logo-notext.svg.png" style="border-radius: 10px;">
<p style="font-size: 0.6em; color: grey;">Python</p>

</div>
</div>

<!--s-->

## Key Features of <span class="code-span">multiprocessing</span>

<div class = "col-wrapper">
<div class="c1" style = "width: 60%">

- **Process-based Parallelism**: Leverages multiple processes to bypass the Global Interpreter Lock (GIL) and achieve true parallelism.
- **Simple API**: Easy to use with constructs like <span class="code-span">Pool</span>, <span class="code-span">Process</span>, and <span class="code-span">Queue</span>.
- **Scalability**: Can scale with the number of available CPU cores, making it suitable for both small and large datasets.

</div>
<div class="c2" style = "width: 30%; text-align: center;">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/640px-Python-logo-notext.svg.png" style="border-radius: 10px;">
<p style="font-size: 0.6em; color: grey;">Python</p>

</div>
</div>

<!--s-->

## Example: Using <span class="code-span">multiprocessing</span> for Data Preprocessing

```python
import multiprocessing as mp
import pandas as pd

# Function to preprocess a chunk of data
def preprocess_chunk(chunk):
  # Example preprocessing steps
  chunk['processed'] = chunk['raw'].apply(lambda x: x * 2)
  return chunk

# Function to split data into chunks and process in parallel
def parallel_preprocess(data, num_chunks):
  chunks = np.array_split(data, num_chunks)
  pool = mp.Pool(processes=num_chunks)
  processed_chunks = pool.map(preprocess_chunk, chunks)
  pool.close()
  pool.join()
  return pd.concat(processed_chunks)

# Example usage
if __name__ == '__main__':
  data = pd.DataFrame({'raw': list(range(1000000))})
  num_chunks = mp.cpu_count()
  processed_data = parallel_preprocess(data, num_chunks)
  print(processed_data.head())
```

<!--s-->

## Best Practices for Using <span class="code-span">multiprocessing</span>

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Chunk Size
Choose an appropriate chunk size to balance between parallelism and the overhead of process management.

### Resource Monitoring
Monitor CPU and memory usage to avoid overloading the system.

</div>
<div class="c2" style = "width: 50%">

### Error Handling
Implement robust error handling to manage exceptions in parallel processes.

### Profiling
Profile the preprocessing pipeline to identify and address bottlenecks.

</div>
</div>

<!--s-->

## Other ~ Local Approaches

### CPUs
- **Dask**: A flexible parallel computing library that provides parallelized data structures and task scheduling.
- **Joblib**: A library that provides simple tools for parallelizing Python functions using <span class="code-span">multiprocessing</span>.
- **PySpark**: A distributed computing framework that provides parallel data processing using Apache Spark, but can be used locally as well.

### CPUs and GPUs
- **Ray**: A distributed computing framework that enables parallel and distributed Python applications. 🔥
- **Numba**: A just-in-time compiler that accelerates Python functions using the LLVM compiler infrastructure.
- **Jax**: A library for numerical computing that provides automatic differentiation and GPU/TPU acceleration.

<!--s-->

<div class="header-slide">

# Distributed Training Strategies

</div>

<!--s-->

<div class="header-slide">

## Data Parallelism

</div>

<!--s-->

## Data Parallelism

Data parallelism is a technique used to distribute the training of a machine learning model across multiple devices, such as GPUs or CPUs. The main idea is to split the training data into smaller batches and process them in parallel on different devices.

<div style="text-align: center;">
  <img src="https://storage.googleapis.com/slide_assets/parallelism.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Scalar Topics (2023)</p>
</div>

<!--s-->

## Advantages and Challenges of Data Parallelism

| Advantages | Challenges |
| --- | --- |
| Scalability (can leverage multiple devices) | Communication overhead (synchronizing gradients) |
| Fault tolerance (if one device fails, others can continue) | Load balancing (ensuring equal workload across devices) |
| Flexibility (can use different devices) | Complexity (requires careful implementation) |
| Improved training speed | Memory constraints (limited by the smallest device) |


<!--s-->

## Pseudocode for Data Parallelism

```
1. Initialize:
  - D: training dataset
  - M: model
  - N: number of devices

2. Split D into N subsets: D1, D2, ..., DN

3. For each device i in {1, 2, ..., N}:
  - Copy model M to device i
  - Train model Mi on data subset Di
  - Compute gradients Gi

4. Aggregate gradients: G = (G1 + G2 + ... + GN) / N

5. Update global model M using aggregated gradients G
```

<!--s-->

## Data Parallelism in Practice

**TensorFlow**: Offers data parallelism with <span class = 'code-span'>tf.distribute.Strategy </span>.
<br>**PyTorch**: Offers data parallelism through <span class = 'code-span'>torch.nn.parallel.DistributedDataParallel</span>.

### TensorFlow Example

```python
import tensorflow as tf

# Define model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # Define strategy
    strategy = tf.distribute.MirroredStrategy()

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

    # Compile and train model within strategy scope
    with strategy.scope():
        model = create_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(dataset, epochs=10)
```

<!--s-->

## Data Parallelism | Best Practices

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Batch Size
Choose an appropriate batch size to balance between computation and communication. For example, a larger batch size can reduce the number of communication rounds.

### Synchronization
Use efficient synchronization techniques to minimize communication overhead. For example, using gradient compression.

</div>
<div class="c2" style = "width: 50%">

### Profiling
Profile the training process to identify and address bottlenecks. For example, use TensorBoard to visualize the training process and identify slow operations.

</div>
</div>


<!--s-->

<div class="header-slide">

# Model Parallelism

</div>

<!--s-->

## Model Parallelism

Model parallelism is a technique used to distribute the training of a machine learning model across multiple devices by splitting the model itself, rather than the data.

<div style="text-align: center;">
  <img src="https://storage.googleapis.com/slide_assets/parallelism.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Scalar Topics (2023)</p>
</div>

<!--s-->

## Types of Model Parallelism

There are several methods to achieve model parallelism.

<div class = "col-wrapper" style = "font-size: 0.8em;">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Layer-wise Parallelism
Splitting model layers across different devices. For example, one group of layers might be placed on one GPU, while the subsequent layers are placed on another. This is often called naive model parallelism, and results in idle GPUs.

### Pipeline Parallelism
Distributes the model across multiple devices in a pipeline fashion, where each device processes a different stage of the model. Through the use of micro-batching, this can help reduce idle time.

</div>
<div class="c2" style = "width: 50%">

### Tensor Parallelism
Distributes individual tensors across multiple devices, often by splitting the tensors themselves (e.g., slices or chunks of matrices/vectors). Commonly used for distributing large matrices involved in operations like matrix multiplication or transformation (e.g., in transformer models).

</div>
</div>


<!--s-->

## Layer-Wise Parallelism vs Pipeline Parallelism Resource Usage

On the top we can see an implementation of layer-wise parallelism, where the GPUs are idle while waiting for the previous GPU to finish processing. 

On the bottom we can see an implementation of pipeline parallelism, where the GPUs are used more efficiently through the use of micro-batching.

<div style="text-align: center;">
  <img src="https://1.bp.blogspot.com/-fXZxDPKaEaw/XHlt7OEoMtI/AAAAAAAAD0I/hYM_6uq2BTwaunHZRxWd7JUJV43fEysvACLcBGAs/s640/image2.png" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Google Research (2019)</p>
</div>

<!--s-->

## Pseudocode for Model Parallelism

```text
1. Initialize:
  - M: model
  - N: number of devices

2. Partition M into N parts: M1, M2, ..., MN

3. Assign each part Mi to device i

4. Forward pass:
  - For each device i in {1, 2, ..., N}:
    - Compute forward pass for Mi
    - Send intermediate results to device i+1

5. Backward pass:
  - For each device i in {N, N-1, ..., 1}:
    - Compute gradients for Mi
    - Send gradients to device i-1

6. Aggregate gradients and update model parameters
```

<!--s-->

## Advantages and Challenges of Model Parallelism

| Advantages | Challenges |
| --- | --- |
| Memory Efficiency: Allows training of very large models that do not fit into the memory of a single device. | Complexity: More complex to implement compared to data parallelism. |
| Scalability: Can leverage multiple devices to speed up training. | Communication Overhead: Requires efficient communication of intermediate results between devices. |
| | Load Balancing: Ensuring that each device has an equal amount of work can be challenging. |

<!--s-->

## Model Parallelism in Practice

The implementation of model parallelism can vary significantly depending on the framework used. Below is an example of model parallelism in TensorFlow.

### Example (TensorFlow)

```python
import tensorflow as tf
import numpy as np

# Define model layers
class ModelPart1(tf.keras.layers.Layer):
  def __init__(self):
    super(ModelPart1, self).__init__() # init parent class.
    self.dense1 = tf.keras.layers.Dense(128, activation='relu')

  def call(self, inputs):
    return self.dense1(inputs)

class ModelPart2(tf.keras.layers.Layer):
  def __init__(self):
    super(ModelPart2, self).__init__()
    self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

  def call(self, inputs):
    return self.dense2(inputs)

# Create model parts and place them on different devices
with tf.device('/GPU:0'):
  model_part1 = ModelPart1()

with tf.device('/GPU:1'):
  model_part2 = ModelPart2()

# Define the full model
class ParallelModel(tf.keras.Model):
  def __init__(self, model_part1, model_part2):
    super(ParallelModel, self).__init__()
    self.model_part1 = model_part1
    self.model_part2 = model_part2

  def call(self, inputs):
    x = self.model_part1(inputs)
    return self.model_part2(x)

# Create the parallel model
parallel_model = ParallelModel(model_part1, model_part2)

# Compile the model
parallel_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create dummy data
x_train = np.random.random((1000, 784))
y_train = np.random.randint(10, size=(1000,))

# Train the model
parallel_model.fit(x_train, y_train, epochs=10, batch_size=32)
```

<!--s-->

## Best Practices for Model Parallelism

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin-right: 2em;">

### Partitioning

Carefully partition the model to balance the workload across devices -- you don't want one device to be idle while another is overloaded.

### Communication

Optimize communication between devices to minimize overhead -- this is done through efficient data transfer and synchronization techniques. Hardware accelerators like NVIDIA NVLink can help.

</div>
<div class="c2" style = "width: 50%">

### Profiling

Profile the training process to identify and address bottlenecks -- this can be done using tools like TensorBoard or PyTorch Profiler.

</div>
</div>

<!--s-->

<div class="header-slide">

# Monitoring and Logging

</div>

<!--s-->

## TensorBoard

TensorBoard is a powerful visualization tool for TensorFlow that allows you to monitor and visualize various aspects of your machine learning model during training. It provides a suite of tools to help you understand, debug, and optimize your model.

<div style="text-align: center;">
  <img src="https://www.tensorflow.org/static/tensorboard/images/tensorboard.gif" width="50%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">TensorBoard (2023)</p>
</div>

<!--s-->

## TensorBoard | Example

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

```python

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
...

# Define model
def create_model():
    ...
    return model

# Create TensorBoard callback
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile and train model
model = create_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])

```

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src = "https://www.tensorflow.org/static/tensorboard/images/tensorboard.gif" width = "100%" style="border-radius: 10px;">
<p style = "font-size: 0.6em; color: grey;">TensorBoard (2025)</p>

</div>
</div>

<!--s-->

## TensorBoard | Features

| Feature | Description | Why is it useful? |
| --- | --- | --- |
| Scalars | Visualize scalar values (e.g., loss, accuracy) over time | Helps track model performance and identify issues |
| Histograms | Visualize the distribution of weights and biases | Helps understand model behavior and identify overfitting / regularization issues |
| Graphs | Visualize the computation graph of the model | Helps understand model architecture and identify bottlenecks |
| Embeddings | Visualize high-dimensional data in lower dimensions | Helps understand data distribution and clustering |
| Images | Visualize images during training | Helps track data augmentation and preprocessing |
| Text | Visualize text data during training | Helps track data preprocessing and augmentation |


<!--s-->

## Weights & Biases

Weights & Biases (W&B) is a popular tool for experiment tracking, model management, and collaboration in machine learning projects. It provides a suite of tools to help you visualize, compare, and share your machine learning experiments.

<div style="text-align: center;">
  <img src="https://help.ovhcloud.com/public_cloud-ai_machine_learning-notebook_tuto_03_weight_biases-images-overview_wandb.png" width="800%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Weights & Biases (2023)</p>
</div>

<!--s-->

## Weights & Biases | Example

Weights & Biases works through a simple REST API that integrates with popular machine learning frameworks like TensorFlow, PyTorch, and Flax. W&B is excellent because of how flexible it is.

<div class = "col-wrapper">

<div class="c1 col-centered" style = "width: 50%; justify-content: start;">

```python
import wandb

wandb.init(config=args)

model = ...  # set up your model

model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
        wandb.log({"loss": loss})
```
<p style="font-size: 0.6em; color: grey;">Weights & Biases (2025)</p>

</div>
<div class="c2 col-centered" style = "width: 50%; justify-content: start;">

<div style="text-align: center;">
  <img src="https://help.ovhcloud.com/public_cloud-ai_machine_learning-notebook_tuto_03_weight_biases-images-overview_wandb.png" width="100%" style="border-radius: 10px;">
  <p style="font-size: 0.6em; color: grey;">Weights & Biases (2023)</p>
</div>

</div>
</div>

<!--s-->

## Weights & Biases | Features

| Feature | Description | Why is it useful? |
| --- | --- | --- |
| Experiment Tracking | Track and visualize experiments, hyperparameters, and metrics | Helps compare and analyze different experiments |
| Model Management | Version control for models and datasets | Helps manage and share models |
| Collaboration | Share experiments and results with team members | Facilitates collaboration and knowledge sharing |
| Integration | Integrates with popular machine learning frameworks | Easy to use with existing projects |

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with multiprocessing, data & model parallelism, and monitoring platforms?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Final Exam

</div>

<!--s-->

## Final Exam Format

- **Format**: Open lecture (you may refer to lecture slides and examples during the exam). **No LLMs or personal notes**. 
- **Duration**: ~ 1 hour.
- **Content**: 5-7 free response questions covering all topics discussed in the course. Questions will be focused on processes and decision making at a high level, rather than the implementation details.

<!--s-->
