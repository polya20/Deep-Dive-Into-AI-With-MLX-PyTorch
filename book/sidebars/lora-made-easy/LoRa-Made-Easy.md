# LoRa Made Easy

## What is LoRa?

LoRa (Low Rank Adaptation) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRa works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRa modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

In order to understand LoRa, we need to understand the concept of rank in matrices, first.

## Ranks and Axes 

You will often encounter the terms 'rank' and 'axis' in the context of arrays, particularly in machine learning and data science as in LoRa. These concepts are closely related to the dimensions of an array, but they're not the same thing. 

Let's clarify the difference.

In mathematics, particularly linear algebra, the rank of a matrix is a fundamental concept that reflects the dimensionality of the vector space spanned by its rows or columns. Here's a simple breakdown:

- Basic Definition: The rank of a matrix is the maximum number of linearly independent column vectors in the matrix or, equivalently, the maximum number of linearly independent row vectors.
- Linear Independence: A set of vectors is linearly independent if no vector in the set is a linear combination of the other vectors. In simpler terms, each vector adds a new dimension or direction that can't be created by combining the other vectors.
- Interpretation: The rank tells you how much useful information the matrix contains. If a matrix has a high rank, it means that it has a large number of independent vectors, indicating a high level of information or diversity.
- Applications: In solving systems of linear equations, the rank of the matrix can determine the number of solutions – whether there's a unique solution, no solution, or infinitely many solutions.

## Examples of Rank in Matrices

### Mathematical Example

Consider the following matrices:

Matrix A:

![matrix-a.png](..%2F..%2F002-adventure-of-tenny-the-tensor%2Fmatrix-a.png)

In Matrix A, the second row is a multiple of the first row (3 is 3 times 1, and 6 is 3 times 2). So, they are not linearly independent. It basically means you get no further information by adding the second row. It's like having two identical rows. Thus, the rank of this matrix is 1. The rank is the answer to a question: "How much useful information does this matrix contain?" Yes, this matrix has only one row of useful information. 

Matrix B:

![matrix-b.png](..%2F..%2F002-adventure-of-tenny-the-tensor%2Fmatrix-b.png)

In Matrix B, no row (or column) is a linear combination of the other. Therefore, they are linearly independent. The rank of this matrix is 2. Why? Because it has two rows of useful information.

### Python Code Example

To calculate the rank of a matrix in Python, you can use the NumPy library, which provides a function `numpy.linalg.matrix_rank()` for this purpose. Note that PyTorch also has a similar function `torch.linalg.matrix_rank()`. In MLX (as of 0.0,7), no equivalent, just yet.

[rank-numpy.py](..%2F..%2F002-adventure-of-tenny-the-tensor%2Frank-numpy.py)

```python
import numpy as np

# Define matrices
A = np.array([[1, 2], [3, 6]])
B = np.array([[1, 2], [3, 4]])

# Calculate ranks
rank_A = np.linalg.matrix_rank(A)
rank_B = np.linalg.matrix_rank(B)

print("Rank of Matrix A:", rank_A)  # Output: 1
print("Rank of Matrix B:", rank_B)  # Output: 2
```

In this Python code, we define matrices A and B as NumPy arrays and then use `np.linalg.matrix_rank()` to calculate their ranks. The output will reflect the ranks as explained in the mathematical examples above.

[rank-torch.py](..%2F..%2F002-adventure-of-tenny-the-tensor%2Frank-torch.py)

In PyTorch:

```python
import torch

# Define a tensor
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute the rank of the tensor
rank = torch.linalg.matrix_rank(A)

# Display the rank
print(rank)
```

[rank-mlx.py](..%2F..%2F002-adventure-of-tenny-the-tensor%2Frank-mlx.py)

In MLX:

```python
import mlx.core as mx

# As of 0.0.7 mlx lacks a rank function

# Define matrices
A = mx.array([[1, 2], [3, 6]], dtype=mx.float32)
B = mx.array([[1, 2], [3, 4]], dtype=mx.float32)

# Function to compute the rank of a 2x2 matrix
def rank_2x2(matrix):
    # Check for zero matrix
    if mx.equal(matrix, mx.zeros_like(matrix)).all():
        return 0
    # Check for determinant equals zero for non-invertible matrix
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if det == 0:
        return 1
    # Otherwise, the matrix is invertible (full rank)
    return 2

# Calculate ranks
rank_A = rank_2x2(A)
rank_B = rank_2x2(B)

print("Rank of Matrix A:", rank_A)  # Output should be 1
print("Rank of Matrix B:", rank_B)  # Output should be 2
```

In MLX, we are using a function to compute the rank of a 2x2 matrix. The function checks for a zero matrix and for a non-invertible matrix. If neither of these conditions is met, the matrix is invertible and has a full rank of 2.

Unfortunately, MLX lacks a rank function, as of 0.0.7. But, we can use the above function to compute the rank of a 2x2 matrix. The function checks for a zero matrix and for a non-invertible matrix. If neither of these conditions is met, the matrix is invertible and has a full rank of 2.

As a matter of fact, here's good use case of LoRa. Most LLMs are well aware of PyTorch, JAX, Tensorflow and other frameworks. But, MLX is a burgeoning framework that came later in the game. So, LLMs are not as aware of MLX as they are of other frameworks. As of this writing, GPT-4 or Copilot have no clue about MLX, still, since they're completion models at core, they pretend to know it and produce code looking very similar to MLX. But, they're not aware of MLX, yet. The code they produce is mumbo jumbo of Python, JAX, PyTorch, Tensorflow, and little bits of MLX. Don't depend on them just yet. 

Theoretically, we can use LoRa to fine-tune LLMs to be more aware of MLX. Surely, given enough amout of good data including thorough documentation and ample examples for LLMs to compare their pretrained knowledge with. This process effectively customizes the LLM to be more aware or knowledgeable about MLX. LoRa's power lies in its ability to adapt and refine a model's capabilities with focused and specialized data, leading to more accurate and contextually aware outputs in areas such as burgeoning fields frameworks like MLX. Basically, they will learn about MLX on top of their existing knowledge of other frameworks.

It's not a perfect example, but take a look at what I'm trying to achieve here: 

[cwk_create_dataset.py](..%2F..%2F..%2Fmlx-examples%2Flora%2Fcwk_create_dataset.py)

By creating a dataset of MLX package docstrings, we can fine-tune LLMs to be more aware of MLX. Theoretically. Some might argue that these kind of data sets are not enough or right to fine-tune LLMs. But here's the deal.

My two cents on the topic:

[The-History-of-Human-Folly.md](..%2F..%2F..%2Fessays%2FAI%2FThe-History-of-Human-Folly.md)

Okay, for math haters. Here goes a simple explanation of the above MLX code:

Think of a matrix like a grid of numbers. Now in MLX, we have written a set of instructions (a function) that can look at a small 2x2 grid – which means the grid has 2 rows and 2 columns.

The function we wrote does a couple of checks:

1. **Check for a Zero Matrix**: The very first thing it does is look to see if all the numbers in the grid are zeros. If they are, then the function says the rank is 0. A "rank" is a way to measure how many rows or columns in the matrix are unique and can't be made by adding or subtracting the other rows or columns. If everything is zero, then there's nothing unique at all. No useful information. The rank is 0.

2. **Check for an Invertible Matrix**: The second thing the function does is a bit like a magic trick. For our 2x2 grid, it performs a special calculation (we call it finding the determinant) to see if the matrix can be turned inside out (inverted). If this special number, the determinant, is zero, then the magic trick didn't work - you can't turn the grid inside out, and the rank is 1. This means there's only one unique row or column. One useful piece of information.

If neither of these checks shows that the matrix is all zeros or that the magic trick failed, then our grid is considered to be fully unique – it has a rank of 2. That's the highest rank a 2x2 grid can have, meaning both rows and both columns are unique in some way.

More dimensions can be added to the grid, and the same checks can be performed. The more dimensions you add, the more checks you need to do. But the idea is the same. If you can't turn the grid inside out, then it's fully unique, and the rank is the highest it can be. If you can turn it inside out, then the rank is lower.

#### Low Rank Adaptation (LoRa)

Low Rank Adaptation (LoRa) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRa works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRa modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

The key benefit of using LoRa is computational efficiency. By reducing the number of parameters that are actively updated, it allows for quicker adaptation of large models to specific tasks or datasets with a smaller computational footprint.

The term "low rank" in this context refers to the property of the matrices that are introduced. A low-rank matrix can be thought of as a matrix that has fewer linearly independent rows or columns than the maximum possible. This means the matrix can be represented with fewer numbers, reducing complexity.

LoRa is particularly useful in scenarios where one wants to customize large AI models for specific tasks (like language understanding, translation, etc.) without the need for extensive computational resources typically required for training such large models from scratch.

In this context, the rank of a matrix is still a measure of its linear independence, but the focus is on leveraging matrices with low rank to efficiently adapt and fine-tune complex models. This approach maintains performance while greatly reducing computational requirements.

For instance, theoretically, with an adequate amount of quality data on a specific topic like MLX, you can fine-tune any capable Large Language Models (LLMs) using that data, thereby creating LoRa (Low-Rank Adaptation) weights and biases. This process effectively customizes the LLM to be more aware or knowledgeable about MLX. LoRa's power lies in its ability to adapt and refine a model's capabilities with focused and specialized data, leading to more accurate and contextually aware outputs in areas such as burgeoning fields frameworks like MLX.

Fine-Tuning LLMs with LoRa examples (from the official Apple repo) are found here:

https://github.com/ml-explore/mlx-examples/tree/main/lora

In Stable Diffusion and similar models, LoRa plays a significant role. For instance, if you have a model adept at creating portraits, applying a LoRa to it can further enhance its capability, specifically tuning it to generate portraits of a particular individual, such as a favorite celebrity. This process is a form of fine-tuning but differs from training a model from scratch. It's more akin to a targeted adaptation, where the model is adjusted to excel in a specific task or with a certain dataset, rather than undergoing a complete retraining. This focused adaptation allows for efficient and effective improvements in the model's performance for specialized applications.