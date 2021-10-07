
Deep Model Watermarking
=======


This repo is implementation for the accepted paper "[Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)" (AAAI 2020) and its extension version "[Deep Model Intellectual Property Protection via Deep Model Watermarking](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9373945&tag=1)" (TPAMI 2021) .


<p align="center"> <img src="./images/pipeline.jpg" width="70%">    </p>
<p align="center"> Figure 1: The overall pipeline of the proposed deep invisible watermarking algorithm and two-stage training strategy. </p>




## How to run

### Initial Training Stage 
Initial 训练阶段主要是训练嵌入子网络H和提取子网络R。
主要是借助于Unet网络结构对水印图像进行嵌入和提取。
该阶段使用了鉴别器D，来识别嵌入水印后的图像的真假，以更好地将水印嵌入到宿主图像中。

Initial Stage阶段的损失都是计算嵌入水印后的图像和B域图像的diff，所以，H不仅有嵌入水印的功能，也有去噪的功能。
也就是说，训练好的H = 原始Imgae-Image任务的模型M + 嵌入水印的模型H'
```
## cd ./Initial stage

python main.py 
```

### Surrogate Model Attack 

这个阶段，我本以为作者会使用Initial阶段得到的带有水印的图像来训练代理模型，但其实这个阶段作者只是在完成Image-Image的Original工作：如derain任务。
具体来说，就是将带有雨线噪声的图像输入到GAN网络中，生成一个不带有雨线的图像，以达到图像去噪的效果。
但是，SM攻击呢？

运行SR stage模型时，batchsz=16只占用了10G左右显存。但是运行Initial Stage阶段，24G显存只能支撑batchsize=4
```
## cd ./SR attack

python train.py 
```

### Adversarial Training Stage 

```
## cd ./Adversarial stage

python main.py 
```

### Watermark Images

```
## We provide some watermark images in the folder "secret".
```

## Experimental Results

<p align="center"> <img src="./images/invisibility.jpg" width="60%">    </p>
<p align="center"> Figure 2: Some visual examples to show the capability of the proposed deep invisible watermarking algorithm. </p>

<p align="center"> <img src="./images/robustness.jpg" width="880%">    </p>
<p align="center"> Figure 3: The robustness of our method resisting the attack from surrogate models. </p>



## Requirements
Python >= 3.6 <br>
Pytorch >= 1.1.0



## Reference By
[arnoweng/PyTorch-Deep-Image-Steganography](https://github.com/arnoweng/PyTorch-Deep-Image-Steganography)<br>
[KupynOrest/DeblurGAN](https://github.com/KupynOrest/DeblurGAN)




## Acknowledgement
This work was supported in part by the
NSFC under Grant 62072421 and 62002334, Exploration Fund
Project of University of Science and Technology of China under Grant YD3480002001, and by Fundamental Research Funds
for the Central Universities under Grant WK2100000011 and
WK5290000001. Jing Liao is partially supported by the Hong
Kong Research Grants Council (RGC) Early Career Scheme under Grant 9048148 (CityU 21209119), and the Shenzhen Basic Research General Program under Grant JCYJ20190814112007258.
Gang Hua is partially supported by National Key R&D Program
of China Grant 2018AAA0101400 and NSFC Grant 61629301.



## Citation
If you find this work useful for your research, please cite
```
@article{zhang2021deep,
  title={Deep Model Intellectual Property Protection via Deep Watermarking},
  author={Zhang, Jie and Chen, Dongdong and Liao, Jing and Zhang, Weiming and Feng, Huamin and Hua, Gang and Yu, Nenghai},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2021},
  publisher={IEEE}
}
```

## License and Copyright
The project is open source under MIT license (see the ``` LICENSE ``` file).

