# moment_compression

## Dependencies

-  Python version <= 3.12
-  FAISS for finding small clusters: install by

```terminal
conda install -c pytorch faiss-cpu
```

## How to

- `demo.py`: visualize 2D or 3D data compressed into atoms. dstop = binom(m+k, k)
- `error_scaling.py`: error scaling for "pretty arbitrary" function defined at the beginning. Results are in `figures/`
- `sine_regression/compress_weights.py`: change params at the beginning of `__main__`. A trained two-layer NN that approximates the sine function is stored in `sine_model.pth`. This code prunes it and compare with the original NN. 