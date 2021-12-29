# Inception-v3 Inference Booster

**Authors**: [Hongjie Fang](https://github.com/galaxies99/), [Peishen Yan](https://github.com/koalayan/), [Haoran Zhao](https://github.com/zhao-hr/).

This is the inference booster of the InceptionV3[1] model. Features includes:

- Implementation of convolution in CPU, CUDA, CUDNN.
- Optimization of convolution (implicit im2col and tilling method).
- Implementation of pooling and FC layer in CPU, CUDA, CUDNN.
- Optimization of the FC layer using tilling method.
- Implementation of the full Inception-v3 network in CPU, CUDA and CUDNN.
- Pytorch inference implementation[2] of Inception-v3 network (only for debug use).
- ONNX-to-JSON formatter for Inception-v3 onnx model.

This is also the final project of course "CS433: Parallel and Distributed Computing" of Shanghai Jiao Tong University, taught by Prof. Xiaoyao Liang.

## Usage

Compile the source codes.

```bash
cd src
make
cd ..
```

You may need to change the `nvcc` path in `src/makefile`. Different compile options are required for different architecture. We only provide compile options for our experiment architecture (Tesla V100, CUDA 10.2).

Download data from [Baidu Netdisk](https://pan.baidu.com/s/1u5jJfNBL9m8prtRMRHuj7Q) (Verify code: csov), and put it in the `data` folder under the root directory of the repository. Then, you can test the inception code using the given model, input and output.

```bash
cd test
./inception_main
cd ..
```

The experiment will run for approximately 10 minutes, which includes 5,000 inference experiments. Here are some experiment statistics.

| Implementation method | Average Inference Time |
| :-: | :-: |
| CPU | ~180,000 ms |
| Our basic CUDA Implementaion | ~36,000 ms |
| CUDNN | 80.851 ms |
| Our CUDA Implementation | **57.424 ms** |

The result show that our implementation is faster than the default implementation of CUDNN.

## Reference

[1] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016;

[2] https://github.com/zt1112/pytorch_inceptionv3/blob/master/inception3.py.