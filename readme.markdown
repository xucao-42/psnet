Code for WACV2020 paper ["PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color"](http://openaccess.thecvf.com/content_WACV_2020/papers/Cao_PSNet_A_Style_Transfer_Network_for_Point_Cloud_Stylization_on_WACV_2020_paper.pdf)
# Introduction
We perform neural style transer on a point cloud from a point cloud or an image from a style point cloud. The geometry or/and the color property of the content point cloud can be stylized. The color propety can also be stylized from an image.
![Teaser](teaser.png)
# Usage
## Dependencies
Code was tested on MacOS 10.14 & above and Ubuntu 16.04 with Python3.6.

Install required packages: `pip3 install -r requirements.txt`.

Be cautious not upgrading matplotlib to 3.1.0 or above, it will drop an error message "It is not currently possible to manually set the aspect on 3D axes" when visualizing point clouds.


## Basic usage

Just run `main.py`. It will style transfer all point clouds in `sample_content` from each style image or point cloud in `sample_style`. Results are saved in a new folder `style_transfer_results`.


## Prepare your data

Put your .ply content point clouds in `sample_content` and your style images or point clouds in `sample_style`. Then run `main.py`.

# Citation
If you used this code in your publication, please consider citing the following paper:
```
@InProceedings{Cao_2020_WACV,
author = {Cao, Xu and Wang, Weimin and Nagao, Katashi and Nakamura, Ryosuke},
title = {PSNet: A Style Transfer Network for Point Cloud Stylization on Geometry and Color},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}
```
# Contact
For any questions/comments/bug reports, please feel free to contact cao.xu@ist.osaka-u.ac.jp