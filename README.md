## DWT_Inpainting: Detail-Enhanced Image Inpainting Based on Discrete Wavelet Transform
Deep-learning-based method has made great breakthroughs in image inpainting by generating visually
plausible contents with reasonable semantic meaning. However, existing deep learning methods still
suffer from distorted structures or blurry textures. To mitigate this problem, completing semantic 
structure and enhancing textural details should be considered simultaneously. To this end, we propose 
a twoparallel-branch completion network, where the first branch fills semantic content in spatial 
domain, and the second branch helps to generate high-frequency details in wavelet domain. To reconstruct 
an inpainted image, the output of the first branch is also decomposed by discrete wavelet transform, and
the resulting low-frequency wavelet subband is used jointly with the output of the second branch. In
addition, for improving the network capability in semantic understanding, a multi-level fusion module
(MLFM) is designed in the first branch to enlarge the receptive field. Furthermore, drawing lessons from
some traditional exemplar-based inpainting methods, we develop a free-form spatially discounted mask
(SD-mask) to assign different importance priorities for the missing pixels based on their positions, 
enabling our method to handle missing regions with arbitrary shapes. Extensive experiments on several
public datasets demonstrate that the proposed approach outperforms current state-of-the-art ones. 
Detailed description of the system can be found in [our paper](https://www.sciencedirect.com/science/article/abs/pii/S0165168421003157). 

## Our framework
<img src="https://github.com/zhengbowei/DWT_Inpainting/tree/main/picture/network.png" width="200" alt="framework"/>

## Result
<img src="https://github.com/zhengbowei/DWT_Inpainting/tree/main/picture/result1.png" width="200" alt="result"/>

## Prerequisites
* Python 3
* PyTorch 1.1.0
* NVIDIA GPU + CUDA cuDNN

## Run
### Training
1. Setting <br>
   Modify parameters in [config.yaml](https://github.com/zhengbowei/DWT_Inpainting/tree/main/configs/config.yaml). <br>
   Eg., <br>
   dataset_name: the name of dataset <br>
   data_with_subfolder: whether the dataset path have subfolders or not <br>
   train_data_path: the path of training dataset
   val_data_path: ignore, do not use
   batch_size: 8
   image_shape: the shape of image after reshapeing
   mask_shape: the shape of missing region
   mask_batch_same: whether a batch of mask have the missing region or not <br>
   spatial_discounting_gamma: the hyperparameter of spatially discounted mask
   random_crop: If ture, the location of missing region will be randomly cropping in the image. If false, the location of missing region will be cropping in the center image.
2. 

## Acknowledgments
The codes are based on [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch) and  [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## Citation
If you use this code for your research, please cite [our paper](https://www.sciencedirect.com/science/article/abs/pii/S0165168421003157).<br>
<br>
@article{li2021detail,<br>
   title={Detail-enhanced image inpainting based on discrete wavelet transforms},<br>
   author={Li, Bin and Zheng, Bowei and Li, Haodong and Li, Yanran},<br>
   journal={Signal Processing},<br>
   volume={189},<br>
   pages={108278},<br>
   year={2021},<br>
   publisher={Elsevier}<br>
}<br>
