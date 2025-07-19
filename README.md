# moment_compression

## To-do

- Write a function reduce1 within the big compressing function
- Future strategy
  - Write an algorithm (e.g., FAISS) that finds a set of N_{m,k}+1 points 
  - reduce1 within these points






### FAISS example

```python
import faiss
import numpy as np

def min_diameter_cluster_faiss(points, N):
    d, m = points.shape
    index = faiss.IndexFlatL2(m)  # Exact L2; swap for approximate if needed
    index.add(points.astype(np.float32))
    _, indices = index.search(points.astype(np.float32), N)
    min_diam = np.inf
    min_cluster = None
    for i in range(d):
        cluster = points[indices[i]]
        dist_matrix = np.linalg.norm(cluster[:, None, :] - cluster[None, :, :], axis=-1)
        diam = np.max(dist_matrix)
        if diam < min_diam:
            min_diam = diam
            min_cluster = indices[i]
    return min_cluster, min_diam
```