def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    Write a Python function that calculates the covariance matrix from a list of vectors. Assume that the input list represents a dataset where each vector is a feature, and vectors are of equal length.

    Example
    Example:
            input: vectors = [[1, 2, 3], [4, 5, 6]]
            output: [[1.0, 1.0], [1.0, 1.0]]
            reasoning: The dataset has two features with three observations each. The covariance between each pair of features (including covariance with itself) is calculated and returned as a 2x2 matrix.
    """
    if len(vectors) == 0:
      return -1
    m, n = len(vectors), len(vectors[0])
    covariance_matrix = [ [0 for _ in range(m)] for _ in range(m)]
    means = [sum(feature)/n for feature in vectors]
    for i in range(m):
      for j in range(m):
        covar = sum( (vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n) ) / (n-1)
        covariance_matrix[i][j] = covariance_matrix[j][i] = covar
    return covariance_matrix