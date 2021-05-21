# Deep Generative Learning via Schrödinger Bridge
Official code for ICML 2021 paper Deep Generative Learning via Schrödinger Bridge.

## Run Experiments
### Training
For the CIFAR-10 dataset, to train both density ratio estimator and score estimator:
```
python main.py --config cifar10.yml --doc cifar10 --sigma_sq 1.0 --tau 2.0
```
To only train density ratio estimator:
```
python main.py --config cifar10.yml --doc cifar10 --sigma_sq 1.0 --tau 2.0 --train_d_only
```
To only train score estimator:
```
python main.py --config cifar10.yml --doc cifar10 --sigma_sq 1.0 --tau 2.0 --train_s_only
```
For the CelebA dataset, the config file is ```celeba.yml```, and recommended hyperparameters are ```--sigma_sq 4.0 --tau 8.0```.
### Sampling
To sample 50,000 images for fid evaluation:
```
python main.py --config cifar10.yml --doc cifar10 --sample --fid
```

## Development
This package is developed by Gefei Wang (gwangas@connect.ust.hk). 

## Contact Information
Please contact Gefei (gwangas@connect.ust.hk), Yuling (yulingjiaomath@whu.edu.cn) or Can (macyang@ust.hk) if any enquiry.

## Reference and Acknowledgements
This implementation is based on https://github.com/ermongroup/ddim.