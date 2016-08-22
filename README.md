# Welcome to the mathCognition_RAM wiki!


## genTrainExp:

In MATLAB, after setting all the parameters appropriately, type `outputImgs()` to generate training images. 

Here're some parameters that you can set: 

* `numImages`: the number of images you want to generate 
* `showImg`: show a sample image after the data is generated 
* `saveImg`: save the data in image format, such as jpg
* `saveStruct`: save the data as feature vector, which is obtained by unrolling the image matrix. At the end of the vector, the number of object (1 element), and the coordinates of the objects (max_num_obj X 2 element) are attached. 
* `pattern`: (string) the distribution of the distortion vector 
* `alignment`: center the object array OR left-align them
* `max_num_obj`: (int) the maximum number of objects in the image 
* `obj_radius`: (int) the radius of the object 
* `varySize`: (bool) randomize the size of the object
* `frame_ver` & frame_hor: (int) vertical and horizontal length of the image 
* `frame_boundary`: (int) distance between objects and the boundary of the image, before distortion  
* `frame_space`: (int) spacing in between each object, before distortion 
* `distortion_x` & `distortion_y`: (float) horizontal and vertical distortion 
* frame_rand_vecDistribution: (string) the shape of the distribution of the distortion vector 
* `supervised`: (bool) indicate that there is a label attached at the end of the image 
* `numObj_sampling`: (string) the distribution of the number of objects in the image 

Here're some examples for **the oneObj data set** (28 x 28 pixels): 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/oneObj.jpg" width="512">


To get these images, first, I place the object at the center, then translate the location of that object by a random vector. The distribution of the random vector is **uniform over a disk** (see the figure below). Note that when the points are uniform over a disk, the norm will not be uniform, since there are more points on the surrounding region (than the center). 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/randVecDistribution_circle_smpSize2000.jpg" width="600">


Similarly, you can generate images with multiple objects, here's a demo:  

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/multiObj.jpg" width="800">

The size of the object can be stochastic: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/multiObj_varySize.jpg" width="800">


There are **two distributions** that you can use (for the random vector): 

1. Elliptical: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/randVecDistribution_elip_smpSize2000.jpg" width="600">

Note that the previous **uniform-disk** is just a special case of this elliptical distribution. 


2. Rectangular: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/randVecDistribution_rect_smpSize2000.jpg" width="600">




## fixate: (under construction)
references: 

[1] https://github.com/seann999/tensorflow_mnist_ram

[2] http://arxiv.org/abs/1406.6247

Fixate on the object with the recurrent attention mechanism. 



## datasets_zipped 

This Git repo hosts several datasets for counting-related tasks:

[1] oneObj (demo above)

[2] oneObj_big 

The objects from the "oneObj" data set were placed on a larger panel (60 x 60). The magnitude of translation was doubled. Here're some example images: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/oneObj_big.jpg" width="800">

[3] multiObj

The amount of objects in each image is randomized (maxNumObj = 5). To capture the environmental regularity, smaller numbers are more likely. Here's the distribution of the random number generator that I wrote: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/numGenDistribution_samp2000.jpg" width="400">

Here're some example images: 

<img src="https://github.com/QihongL/mathCognition_RAM/blob/master/demo/multiObj_randNum.jpg" width="800">
