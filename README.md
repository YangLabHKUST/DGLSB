# Deep Generative Learning via Schrödinger Bridge
Official code for ICML 2021 paper [Deep Generative Learning via Schrödinger Bridge](https://proceedings.mlr.press/v139/wang21l), by Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang and Can Yang.

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

## Citation
```
@InProceedings{pmlr-v139-wang21l,
  title = 	 {Deep Generative Learning via Schr{ö}dinger Bridge},
  author =       {Wang, Gefei and Jiao, Yuling and Xu, Qian and Wang, Yang and Yang, Can},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {10794--10804},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR}
}
```
