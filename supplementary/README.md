# Moment matching compression demo code

## Dependencies

-  Python version <= 3.12
-  PyTorch: CPU or CUDA version
-  FAISS for finding small clusters: install by

```terminal
conda install -c pytorch faiss-cpu
```

## How to

- `compressor.py` Defines the compressor class. Use it as
```python
from compressor import Compressor

cp = Compressor(data, index_type = 'flat' or 'ivf')
weights, compressed_vecs = cp.compress(k, dstop=..., **kwargs)
```
- `demo.py`: visualize 2D or 3D data compressed into weighted objects. dstop = binom(m+k, k)
- `compress_dynamics.py`: a small-scale version of Fig. 4 on the dynamical lottery ticket hypothesis shown in the main text. Easy setting of hyperparameters in lines 137-183. 
- `trainds.py`: a small-scale version of Fig. 3 in the main text, demonstrating compression of training dataset. Easy setting of hyperparameters in lines 103-118. 