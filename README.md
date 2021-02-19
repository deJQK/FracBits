# Network Quantizing with Fractional Bitwidths

This is the codebase of searching bit-widths for neural networks based on fractional bit-widths, as proposed in FracBits [arXiv](https://arxiv.org/abs/2007.02017) [AAAI2021](NA).


## Run

0. Requirements:
    * python3, pytorch 1.0, torchvision 0.2.1, pyyaml 3.13.
    * Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
1. Training and Testing:
    * The codebase is a general ImageNet training framework using yaml config under `apps` dir, based on PyTorch.
    * To test, download pretrained models to `logs` dir and directly run command.
    * To train, comment `test_only` and `pretrained` in config file. You will need to manage [visible gpus](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) by yourself.
    * Command: `python train.py app:{apps/***.yml}`. `{apps/***.yml}` is config file. Do not miss `app:` prefix.
2. Still have questions?
    * If you still have questions, please search closed issues first. If the problem is not solved, please open a new.


## Technical Details

Implementing network quantizing is straightforward:
  * Quantization layers are implemented in [`models/quantizable_ops`](/models/quantizable_ops.py).
  * Training with quantizing is implemented by setting a reasonable [`kappa`] in the yml file.
  * [`q_mobilenetv1_uint8_train_val.yml`] is a good start yml example. For ablation test, please run [`test_ablation`](/test_ablation.sh) with the corresponding test ablation yml file.


## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.


## Citing
```
@article{yang2020fracbits,
  title={FracBits: Mixed Precision Quantization via Fractional Bit-Widths},
  author={Yang, Linjie and Jin, Qing},
  journal={arXiv preprint arXiv:2007.02017},
  year={2020}
}
```
