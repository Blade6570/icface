## ICface: Interpretable and Controllable Face Reenactment Using GANs ##

[**Project**](https://tutvision.github.io/icface/)   

This is the part of implementation for the  "ICface: Interpretable and Controllable Face Reenactment Using GANs" (https://arxiv.org/abs/1904.01909). 

*The test code is released now!*

**Prerequisites**
1. Python 3.5.4
2. Pytorch 0.4.1
3. Visdom and dominate 
4. Natsort

*The code is tested on Ubuntu 16.04 LTS* 

**Download the Pretrained Weights of ICface**
[Google Drive Link](https://drive.google.com/drive/folders/1jvD8R-Ggo3Seg2tF-JsqlOjwropVwN1S?usp=sharing). 

**Testing ICface**

1. Clone the ICface repository and change the working directory to *'/test_code_released'* 
2. Keep the pretrained weights inside the path: *./checkpoints/gpubatch_resnet*. 
3. For the driving video, you can select any video file from *voxceleb* dataset, extract the action units in a .csv file using Opecface and store the .csv file in the working folder. We have provided two such .csv files and thier corresponding driving videos.
4. For the source image, we have selected images from *voxceleb* test set. Three exampes are given in the folder *./new_crop*. More can be obtained from [here](http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/). In particular the "Cropped Face Images extracted at 1fps" (7.8Gb). The test identities can be downloaded <a href="http://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/x2face.html">here</a> under the data section.
5. Run in terminal : *python test.py --dataroot ./ --model pix2pix --which_model_netG resnet_6blocks --which_direction AtoB --dataset_mode aligned --norm batch --display_id 0 --batchSize 1 --loadSize 128 --fineSize 128 --no_flip --name gpubatch_resnet --how_many 1 --ndf 256 --ngf 128 --which_ref ./new_crop/1.png --gpu_ids 1 --csv_path 00116.csv --results_dir results_video*
6. The resuting video will be found in *'/test_code_released'* under the name *'movie.mp4'*

**If you are not using *voxceleb* test set**
1. In the python file *'image_crop.py'*, add your image path and run it.
2. It will create a new cropped version of your image and will store in *'./new_crop'* folder. Then follow the above steps to create youe video file. 

**_If you are using this implementation for your research work then please cite us as:_**
 
```
#Citation 

@article{tripathy+kannala+rahtu,
  title={ICface: Interpretable and Controllable Face Reenactment Using GANs},
  author={Tripathy, Soumya and Kannala, Juho and Rahtu, Esa},
  journal={arXiv preprint arXiv:1904.01909},
  year={2019}
}

```
