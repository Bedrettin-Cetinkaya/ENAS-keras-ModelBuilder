# ENAS-keras Fixed Architecture Builder
- This project is to implement enas fixed architecture model in keras. It doesn't include whole training code. It only creates  python script of fixed architecture model.
- Cifar10 micro and ptb architecture are in progress.
- Cifar10 macro architecture is completed. 

## Prerequisites
- Keras
- Numpy

## Run 
bash run.sh

- You can find example output script , Model.py, in output folder.
- To obtain enas fixed architecture, you can use pytorch,tensorflow or keras implementations.

## Differences from Original Implementation
- Bias is used.
- In the pooling stage, padding and cropping operations are not used.

## Complete implementations of ENAS
- [Tensorflow](https://github.com/melodyguan/enas)
- [Pytorch](https://github.com/carpedm20/ENAS-pytorch)
- [Keras](https://github.com/shibuiwilliam/ENAS-Keras)

## Reference
- [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268)
