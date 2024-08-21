# LayerNorm in DiT

In DiT architecture, there is a LayerNorm before every layer. LayerNorm destroy the scale and shift information in its input.
However, we posit that this information is useful in the denoising task and should be preserved.

This repository aim to explore this hypothesis. It is a fork from the official DiT repo https://github.com/facebookresearch/DiT/tree/main
The only addition so far is the latent visualization notebook LayerNormViz.ipynb that gives an empyrical evidence that a trained DiT does try to preserve scale and shift information despite the normalization layers.