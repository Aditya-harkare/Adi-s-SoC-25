import numpy as np
def normalize_array(arr):
  min_val = np.min(arr)
  max_val = np.max(arr)
  numerator = arr - min_val
  denominator = max_val - min_val

  normalized_array = numerator / denominator
  return normalized_array

sample_array = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])

normalized_array = normalize_array(sample_array)
print("Original Array:")
print(sample_array)
print("\nNormalized Array:")
print(normalized_array)