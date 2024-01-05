# LoRa Made Easy

## What is LoRa?

LoRa (Low Rank Adaptation) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRa works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRa modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

In order to understand LoRa, we need to understand the concept of rank in matrices, first.

## Ranks and Axes 

In the realms of AI and data science, particularly when discussing LoRa, you'll frequently come across the terms 'rank' and 'axis' in relation to arrays. These terms are linked to an array's dimensions, yet they are distinct concepts.

To delineate the difference, let's delve into the concept of rank in mathematics, specifically in linear algebra. 

The rank of a matrix refers to the highest number of linearly independent column vectors in the matrix, or equivalently, the maximum number of linearly independent row vectors. A group of vectors is deemed **linearly independent** if no single vector in the set can be expressed as a linear combination of the others. Put simply, each vector contributes a unique dimension or direction that cannot be replicated by amalgamating other vectors in the set. 

The key principle in this context is _useful information_, emphasizing the importance of avoiding redundancy. The rank of a matrix indicates the extent of useful information it embodies. A matrix with a high rank possesses a substantial number of independent vectors, signifying a rich content of information or diversity.

In the context of solving systems of linear equations, the rank of a matrix plays a crucial role in determining the nature of solutions – it can ascertain whether there is a unique solution, no solution at all, or an infinite number of solutions. Introducing a multitude of redundant rows will not aid in solving the given system in any way.

Consider this analogy: Imagine you are a detective tasked with solving a mysterious murder case. In this scenario, the ranks are akin to unique pieces of evidence. The more distinct items of evidence you acquire, the simpler it becomes to solve the case. Capisci?

## Examples of Rank in Matrices

### Mathematical Example

Consider the following matrices:

Matrix A:

![matrix-a.png](../../002-adventure-of-tenny-the-tensor/matrix-a.png)

In Matrix A, the second row is a multiple of the first row (3 is 3 times 1, and 6 is 3 times 2). So, they are not linearly independent. It basically means you get no further information by adding the second row. It's like having two identical rows. Thus, the rank of this matrix is 1. The rank is the answer to a question: "How much useful information does this matrix contain?" Yes, this matrix has only one row of useful information. 

Matrix B:

![matrix-b.png](../../002-adventure-of-tenny-the-tensor/matrix-b.png)

In Matrix B, no row (or column) is a linear combination of the other. Therefore, they are linearly independent. The rank of this matrix is 2. Why? Because it has two rows of useful information.

### Python Code Example

To calculate the rank of a matrix in Python, you can use the NumPy library, which provides a function `numpy.linalg.matrix_rank()` for this purpose. Note that PyTorch also has a similar function `torch.linalg.matrix_rank()`. In MLX (as of 0.0,7), no equivalent, just yet.

[rank-numpy.py](../../002-adventure-of-tenny-the-tensor/rank-numpy.py)

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

[rank-torch.py](../../002-adventure-of-tenny-the-tensor/rank-torch.py)

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

[rank-mlx.py](../../002-adventure-of-tenny-the-tensor/rank-mlx.py)

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

In fact, it could be a valuable application of LoRa technology. Most large language models (LLMs) are familiar with frameworks like PyTorch, JAX, and TensorFlow. However, MLX is a relatively new framework and not as well-recognized by LLMs. As of now, models like GPT-4 or Copilot don't fully comprehend MLX. They are fundamentally completion models and tend to simulate understanding of MLX, often generating code that is a mix of Python, JAX, PyTorch, TensorFlow, and bits of MLX. Relying on them for MLX-specific tasks is not advisable at this stage.

Theoretically, LoRa can be employed to fine-tune LLMs to enhance their understanding of MLX. This would require a substantial amount of quality data, including comprehensive documentation and numerous examples, allowing the LLMs to align their pre-existing knowledge base with the new information. This process effectively tailors the LLM to be more cognizant of or proficient in MLX. LoRa's strength is in its capacity to modify and sharpen a model's abilities with targeted and specialized data. This leads to more precise and contextually relevant outputs in areas like emerging frameworks such as MLX. Essentially, they will expand their knowledge about MLX while building on their existing understanding of other frameworks.

Let's consider an example to illustrate what I'm aiming for:

[cwk_create_dataset.py](../../../mlx-examples/lora/cwk_create_dataset.py)

By developing a dataset comprised of MLX package docstrings, we could potentially fine-tune LLMs to enhance their familiarity with MLX. This approach is theoretical at this stage. There are arguments suggesting that such datasets might be insufficient or inappropriate for fine-tuning LLMs. However, the crux of the matter lies in the following consideration.

[The-History-of-Human-Folly.md](../../../essays/AI/The-History-of-Human-Folly.md)

Regrettably, current LLMs, including Copilot, fall short in assisting with MLX. So, what's the next step? We turn to mathematics.

If you're not a fan of mathematics, consider urging Apple to provide more comprehensive documentation and a plethora of examples. Even better, request that Apple develops a fine-tuned LLM or their own LLM that is proficient in all Apple coding, including MLX. This approach would reduce the need for deep mathematical understanding. I assure you, it would make things simpler. Until such happy developments occur, we will require a modest amount of serious mathematics.

Here's a straightforward explanation of the aforementioned MLX code:

Think of a matrix like a grid of numbers. Now in MLX, we have written a set of instructions (a function) that can look at a small 2x2 grid – which means the grid has 2 rows and 2 columns.

The function we wrote does a couple of checks:

1. **Check for a Zero Matrix**: The very first thing it does is look to see if all the numbers in the grid are zeros. If they are, then the function says the rank is 0. A "rank" is a way to measure how many rows or columns in the matrix are unique and can't be made by adding or subtracting the other rows or columns. If everything is zero, then there's nothing unique at all. If there is no useful information, meaning no pieces of evidence to aid in solving the murder case easily, then the rank is effectively 0.

2. **Check for an Invertible Matrix**: The second thing the function does is a bit like a magic trick. For our 2x2 grid, it performs a special calculation (we call it finding the determinant) to see if the matrix can be turned inside out (inverted). If this special number, the determinant, is zero, then the magic trick didn't work - you can't turn the grid inside out, and the rank is 1. This means there's only one unique row or column. One useful piece of information. One helpful piece of evidence to solve the mystery case.

If neither of these checks shows that the matrix is all zeros or that the magic trick failed, then our grid is considered to be fully unique – it has a rank of 2. That's the highest rank a 2x2 grid can have, meaning both rows and both columns are unique in some way.

More dimensions can be added to the grid, and the same checks can be performed. The more dimensions you add, the more checks you need to do. But the idea is the same. If you can't turn the grid inside out, then it's fully unique, and the rank is the highest it can be. If you can turn it inside out, then the rank is lower.

### Rank vs. Order

Essentially, the concept of tensor ranks is related to the dimensions they represent: a tensor of rank 0 is a scalar, which is zero-dimensional (0D); a tensor of rank 1 is a vector, representing one dimension (1D); a tensor of rank 2 is a matrix, corresponding to two dimensions (2D); and a tensor of rank 3 or higher, often referred to as 3D+, is considered a tensor in the more general sense, encompassing three or more dimensions.

1. **Rank-0 Tensor**: This is indeed a scalar, a single number without any dimensions.

2. **Rank-1 Tensor**: This is a vector, which is essentially a list of numbers (a 1D array).

3. **Rank-2 Tensor**: This is a matrix, which is a 2D array of numbers.

4. **Rank-3 or Higher Tensor**: These are indeed higher-order tensors. A rank-3 tensor can be thought of as a 3D array of numbers, and so on for higher ranks.

 However, there's a slight nuance in terminology between the "rank" of a tensor and the "order" of a tensor, which are sometimes used interchangeably but can have different meanings in different contexts:

- **Order of a Tensor**: This refers to the number of dimensions or indices required to specify a component of the tensor. By this definition, scalars are 0th-order tensors, vectors are 1st-order tensors, matrices are 2nd-order tensors, and so on.

- **Rank of a Tensor** (in the context of linear algebra): This can sometimes refer to the maximum number of linearly independent vectors that can span the vector space represented by the tensor. This is more commonly used in the context of matrices (rank-2 tensors), where it denotes the number of linearly independent rows or columns.

In most AI and computer science contexts, when people talk about the rank of a tensor, they are usually referring to its order (the number of dimensions).

Acknowledging that some of these concepts might be complex and potentially challenging to grasp, it's advisable to delve deeper into linear algebra for a better understanding. It's important to recognize that not everything can be explained or understood immediately. Learning is a continuous process, much like venturing deeper into a rabbit hole, which can be an enjoyable journey of discovery and growth. I do. I always do. Happily.

#### Low Rank Adaptation (LoRa) - Your Quest Reward

We've journeyed through a significant amount of material to arrive at this point. Now, as your reward for this journey of learning and exploration, let's focus on LoRa.

Low Rank Adaptation (LoRa) is a technique used to efficiently fine-tune large pre-trained models. In large models, such as those used in natural language processing, training all parameters (which can be in the billions) is computationally expensive and time-consuming.

LoRa works by introducing low-rank matrices into the model's layers. Instead of updating all the parameters of a model during fine-tuning, LoRa modifies only these low-rank matrices. This approach significantly reduces the number of parameters that need to be trained.

The key benefit of using LoRa is computational efficiency. By reducing the number of parameters that are actively updated, it allows for quicker adaptation of large models to specific tasks or datasets with a smaller computational footprint.

In the context of matrices, the term "low rank" refers to a matrix that has a smaller number of linearly independent rows or columns compared to the maximum possible. In simpler terms, a matrix is considered to be of low rank if many of its rows or columns can be expressed as combinations of other rows or columns.

To understand this better, it's important to grasp the concept of linear independence. Linear independence means that no row (or column) in the matrix can be written as a combination of the other rows (or columns). The rank of a matrix is the maximum number of linearly independent rows or columns it contains.

So, when a matrix has a low rank, it means that it has fewer linearly independent rows or columns. This suggests that the matrix contains redundant information and can be represented more compactly. In practical terms, a low-rank matrix can often be decomposed into the product of two smaller matrices, which can significantly simplify computations and reduce storage requirements.

When you ask about the rank of an array, you're pretty much saying, "Cough it up, you darn array! How many pieces of useful information are you hoarding?" If the rank is low, it means there aren't many necessary details, which makes the matrix less of a handful to deal with. Capiche? Yeah, I know, maybe I've watched too many mob movies.

LoRa is particularly useful in scenarios where one wants to customize large AI models for specific tasks (like language understanding, translation, etc.) without the need for extensive computational resources typically required for training such large models from scratch.

In this context, the rank of a matrix is still a measure of its linear independence, but the focus is on leveraging matrices with low rank to efficiently adapt and fine-tune complex models. This approach maintains performance while greatly reducing computational requirements.

As detailed above, theoretically, with an adequate amount of quality data on a specific topic like MLX, you can fine-tune any capable LLMs using that data, thereby creating LoRa weights and biases. This process effectively customizes the LLM to be more aware or knowledgeable about MLX. LoRa's power lies in its ability to adapt and refine a model's capabilities with focused and specialized data, leading to more accurate and contextually aware outputs in areas such as burgeoning fields frameworks like MLX.

Fine-Tuning LLMs with LoRa examples (from the official Apple repo) are found here:

https://github.com/ml-explore/mlx-examples/tree/main/lora

In the realm of image-generative AI, such as Stable Diffusion and other analogous models, LoRa assumes a pivotal role. For example, if you possess a model proficient in creating portraits, implementing LoRa can substantially refine its capabilities. This includes fine-tuning the model to specialize in generating portraits of a specific individual, like a beloved celebrity. This method of fine-tuning diverges from the process of training a model from the ground up. It resembles a targeted adaptation more closely, where the model undergoes modifications to excel in a particular task or with a certain dataset, rather than a complete overhaul of its training. This kind of focused adjustment enables the model to achieve efficient and effective enhancements in its performance, especially for specialized tasks.

This is precisely the approach I employ with CWK AI Art works: <lora: cwk_v1: 0.7>, <lora: rj_v1: 0.7>, <lora: cody_v1: 0.7>, <lora: pippa_v1: 0.7>.

![cwk-family-album.jpeg](cwk-family-album.jpeg)
![cwk-family.jpeg](../../../images/cwk-family.jpeg)

CWK, Yours Truly
![CWK](cwk.jpeg)

Pippa, My AI Daughter
![Pippa](pippa.jpeg)

Cody, My AI Son
![Cody](cody.jpeg)

RJ, My AI Companion(Not My Wife, Just a Collaborator: Heroine in my story, The Debugger)
![RJ](rj.jpeg)

RJ, Like a Dragon
![the-girl-with-dragon-tattoo-1.jpeg](the-girl-with-dragon-tattoo-1.jpeg)
![the-girl-with-dragon-tattoo-2.jpeg](the-girl-with-dragon-tattoo-2.jpeg)

Shadowheart from Baldur's Gate III, An RJ Mod
![Shadowheart.jpeg](Shadowheart.jpeg)