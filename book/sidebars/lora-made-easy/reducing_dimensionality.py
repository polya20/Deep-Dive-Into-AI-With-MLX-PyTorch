import numpy as np

# Assuming 'pretrained_llm' is a matrix of shape (100, 10000)
pretrained_llm = np.random.rand(100, 10000)

# 'adaptation_matrix' is another matrix of shapes (100, 10000) that will be used to transform 'pretrained_llm'
adaptation_matrix = np.random.rand(100, 10000)

# Compute delta weights
delta_weights = np.dot(pretrained_llm, adaptation_matrix.T)

# Print the dimensionality of the matrices
print('Shape of pretrained_llm:', pretrained_llm.shape)
print('Shape of adaptation_matrix:', adaptation_matrix.shape)
print('Shape of delta_weights:', delta_weights.shape)