# moment_compression

## Dependencies

-  Python version <= 3.12
-  FAISS for finding small clusters: install by

```terminal
conda install -c pytorch faiss-cpu
```

## How to

- `compressor.py` use the following code to implement data compression
```python
from compressor import Compressor

cp = Compressor(data, index_type = 'flat' or 'ivf')
weights, compressed_vecs = cp.compress(k, dstop=..., **kwargs)
# cp.compress_weights(k, **kwargs) can output weights at various intermediate stages
```
- `demo.py`: visualize 2D or 3D data compressed into atoms. dstop = binom(m+k, k)
- `error_scaling.py`: error scaling for "pretty arbitrary" function defined at the beginning. Results are in `figures/`
- `sine_regression/compress_weights.py`: change params at the beginning of `__main__`. A trained two-layer NN that approximates the sine function is stored in `sine_model.pth`. This code prunes it and compare with the original NN. 

## To do

- Error scaling for sine regression
- A direct showcase/scaling for the lottery ticket hypothesis: Learning something with d=10000. Compress to d'=0.1. Start over with these "WEIGHTED" neurons. Train with the same hyperparams. Compare output. 