---
layout: post
title:  "A Survey of Deep Learning Methods for 3D Object Classification"
date:   2020-12-18 18:05:55 +0300
image:  '/assets/img/data.jpg'
tags:   CV 
---
# Abstract
The objective of this study was to survey different deep learning techniques to analyze 3D data for object classification. Understanding 3D data is attracting attention due to its various applications from autonomous driving to medical image processing. Unlike 2D data, which has been extensively researched, traditional classification techniques on 3D data lack robustness. However, due to recent advances and lower costs, 3D data is gaining abundance and driving innovation in the 3D space. The various approaches surveyed in this study all revolve around the utilization of deep learning algorithms. These techniques are based on different 3D data representations such as multi-view RGB-D data, volumetric data, and point cloud data. This study implements a convolutional neural network on a 3D point cloud dataset of handwritten digits. The same convolutional neural network architecture is then compared to a similar 2D version of the dataset to compare accuracy results. The goals of the study are to understand the recent advancements and current state of 3D object recognition.

# Introduction
Deep learning has proven itself to be a very powerful tool for developing machine learning models in both one and two dimensions. In one-dimension, deep learning is applied to time-series data to model a broad range of applications from language to finance. Similarly, in two-dimensions, deep learning has been extensively researched in images where a 2D matrix of an image is passed to a deep learning model. One of the key strengths of deep learning is its ability to discriminate, detect, and correlate features on its own which enables faster learning without explicit supervision. As a result, this powerful tool has been extensively researched and is already established on 2D data. 

Nevertheless, we live in a 3D world and increasing attention is being placed on deep learning techniques to model various aspects of the world around us. Historically, most 3D models lacked robustness due to a shortage of 3D data. Due to this reason, applying deep learning techniques on 3D data was not as effective as on traditional 1D or 2D domains. However, recent development of cameras and scanning devices, like Intel’s LIDAR camera and Microsoft’s Kinect sensor, are leading development in 3D deep learning by making data easier to obtain. The surge in 3D deep learning started in 2015 along with the distribution of key 3D datasets: ShapeNet and ModelNet. The introduction of numerous open source 3D datasets was the impetus behind research to some of the most complex 3D problems. In autonomous driving, deep learning is utilized extensively by the automotive industry to model perception. Similarly, deep learning is used by medical practitioners in 3D medical image processing. 

While there are many applications for 3D deep learning they usually fall into one of three tasks: 3D geometry analysis, 3D-assisted image analysis, and 3D synthesis. 3D geometry analysis encompasses the realms of object classification, parsing, and correspondence. 3D synthesis involves the reconstruction of objects from 2D images, shape completion, and shape modeling. 3D-assisted images analysis deals with image retrieval and decomposition. These various subfields of 3D deep learning rely on different data representations. In particular, this study focuses on multi-view representations, volumetric representations, RGB-D data, and point clouds for object classification.

# Multi-View Data
Multi-view data is represented as a combination of 2D images captured from multiple points of view. These images are used to capture the 3D object as a whole. Deep learning is applied to each point of view individually and features are extracted. Learned features are then merged together through view pooling. View-pooling is used to process multiple views for the 3D object in no specific order. The output of view-pooling is then passed through a deep learning neural network for a final prediction of our 3D object. Through back-propagation, we can fine tune our network parameters further. 

## MVCNN
According to Su et al. in [1], building deep learning classifiers for 3D models upon 2D image renderings dramatically outperformed classifiers built directly on 3D representations. One reason for the outperformance is that 3D representations rely on a significant reduction in resolution. On the other hand, 2D image renderings can be trained with deep learning quickly without the need for changes in resolution. Su et al. applied a multi-view convolutional neural network on ModelNet40, a comprehensive dataset of 3D CAD models of objects, and found that their model outperformed traditional 3D models. Their multi-view convolutional neural network (MVCNN) was able to achieve 89.9% accuracy with 12 views and 90.1% accuracy with 80 views for classification tasks. This is in contrast to the 3D ShapeNet’s algorithm developed by Wu et al. in [3] which only achieved a 77.3% classification accuracy.

Another advantage of a MVCNN is the ubiquity of 2D image datasets. Deep learning models can learn a great deal of generic features from 2D image renderings and then be fine-tuned for specific classification problems on 3D projections. Unlike 2D images, large labeled 3D datasets are not abundant and therefore suffer as a result in terms of accuracy. Moreover, 2D image renderings allow for transformations to reduce noise, occlusion, and illumination. 

One concern with the multi-view representation is the optimal number of views or 2D rendered images that should be used to build the deep learning classifier. Too few views could lead to inaccurate results, as the views don't make up a true representation of the 3D object. In contrast, too many views cause excessive computational overhead and could lead to an overfitted model. Multi-view representations add another parameter that must be optimized in order to get accurate classification results.

## MVCNN with VGG-11
Su et al. [2] built off of the MVCNN in [1] with a modification to the rendering technique and deeper architectures. Their algorithm improved the accuracy of the MVCNN to 95% per-instance accuracy on the ModelNet40 dataset. One reason for their algorithms' strong performance is due to pretraining. To model the accuracy results of pretraining of MVCNN, Su et al. [2] measured the accuracy of two VGG-11 architectures with and without ImageNet pretraining. VGG or Visual Geometry Group is a pretrained deep convolutional neural network used for large-scale image recognition tasks. VGG-11 with ImageNet pretraining achieved a per instance accuracy of 95% compared to 91.3% without ImageNet pretraining. Su et al. [2] also implemented shaded image renderings for stronger accuracy results. Shaded image renderings were found to achieve 3.4% higher accuracy results compared to depth images and 1.4% accuracy improvement compared to binary silhouette images.

Despite the need to optimize the number of views, multi-view data can be sufficiently used to classify 3D objects. They are suitable to model rigid 3D objects and allow for generalization towards other 3D features. In addition, multi-view data proves more effective in learning 3D features than volumetric representations such as ShapeNet.

<p align="center">
  <img src="/assets/img/Table1.PNG" />
</p>

# Volumetric Data
Another way 3D data can be represented is by a regular grid in 3D space. Just how pixels are the smallest element of a 2D image rendering, a voxel is a basic 3D unit that characterizes the volume in three dimensions. The process of converting geometric objects into a set of voxels is called voxelization. Voxelation can be used to create 3D computer aided design (CAD) models, model the brain in three dimensions from CT and fMRI scans, and model sediment layers for geology. The voxelized object is then run through a 3D convolutional neural network or volumetric convolution. 3D convolutions utilize a kernel that moves in 3 directions (x, y, z) with input and output data in four dimensions. 

As one can imagine, running 3D convolutions with four dimensional kernels can be computationally expensive. A more efficient volumetric approach is octree generating networks. Coined by Tatarchenko et al. in [5], Octree-based networks model 3D objects as a hierarchical structure. A quadtree structure recursively decomposes voxels to divide the scene into cubes that are inside or outside the 3D object. This approach is better able to capture fine details of a 3D object with less computational intensity than a voxel-based approach.

## ShapeNet
One of the very first algorithms to exploit the geometry of voxels was ShapeNet. Developed by Wu et al. in [3], ShapeNet models 3D objects as a probability distribution of binary variables on a 3D voxel grid. ShapeNet utilizes a Convolutional Deep Belief Network, a type of deep learning neural network that was adopted from 2D deep learning to model 3D objects, to represent the probability distribution of binary variables. ShapeNet consists of five layers: an input layer, 3 convolutional layers, and an output layer. These layers are pre-trained using multiple methods. Despite ShapeNet’s simplistic strategy to classify 3D objects based on volumetric data, it suffers from memory storage limitations and information loss. Due to the additional dimension of the convolutional layer, ShapeNet cannot process large amounts of high-resolution data. In contrast, if the resolution is too low, more points fall into a voxel, which increases the information loss. Therefore, finding an optimal resolution is critical to minimize the information loss and computational time of volumetric-based algorithms.

## VoxNet
Building upon the concept of 3D convolutions, Maturana and Scherer in [4] developed VoxNet. The VoxNet architecture consists of an input layer which can take different 3D data representations, like RGB-D, LIBAR point clouds, and 3D CAD models, as input. The input layer is constructed as a volumetric occupancy grid of 32x32x32 voxels. VoxNet then runs the input through two convolutional layers, a pooling layer, and two fully connected layers. In comparison to ShapeNet, VoxNet outperforms ShapeNet on the ModelNet40 dataset with an accuracy of 83% compared to ShapeNet’s accuracy of 77.3%.

## LightNet
In light of the huge computational requirements of volumetric-based approaches, Zhi et al. proposed LightNet in [6]. LightNet is a lightweight 3D convolutional neural network for real-time 3D object recognition. Based on VoxNet, LightNet leverages multitask learning to improve accuracy and efficiency by learning multiple features at the same time. Moreover, LightNet utilizes a batch normalization operation between the convolutional and activation operations to achieve faster convergence. In terms of accuracy, LightNet trained from scratch achieved an accuracy of 82.9% on ModelNet40. This score increases dramatically if LightNet is pretrained on ModelNet10 with an accuracy of 86.9% on ModelNet40. In addition to increased accuracy, LightNet is able to classify a 3D object within 5ms. This means that LightNet can be utilized for real-time object recognition tasks.

While volumetric-based representations like ShapeNet, VoxNet, and LightNet are simple, they struggle in terms of accuracy and computation time compared to multi-view representations. Despite their limitations, volumetric-based approaches prove to be effective for real-time object recognition tasks where recognition time is of the essence.

<p align="center">
  <img src="/assets/img/Table2.PNG" />
</p>

# RGB-D Data
RGB-D image representations are gaining popularity due to the proliferation of novel RGB-D sensors and cameras like Microsoft’s Kinect due to their low costs and simplicity. As the name suggests, RGB-D data provides color images and corresponding depth images that are color, illumination, rotation, and scale invariant. RGB-D datasets can be utilized for many different applications from object classification, scene classification, pose/gesture classification, and much more.

Traditional 2D RGB image renderings only provide information on an object’s appearance. While computer vision techniques exist to decipher common variations in foreground and background, RGB-D images provide a wealth of information to quickly solve these limitations of traditional RGB image renderings. The core of RGB-D data lies in the depth image. This depth image is generated from a range sensor that is robust to variations in in color, illumination, rotation, and scale.

On November 4th, 2010, Microsoft released their Kinect sensor. The Kinect sensor was originally used to eliminate the game controller from Xbox video games. However, the Kinect was soon adopted by academia as a cheaper means to obtain RGB-D images. Due to its low costs and simplicity, RGB-D datasets are widely available compared to other 3D datasets such as point clouds.

## Two-Stream CNN
One of the first deep learning architectures applied to RGB-D data was developed by Socher et al. in [7]. The authors proposed a two-stream convolutional neural network architecture that utilizes both convolutional and recursive layers for object classification. The first stream of RGB and depth data is run through a convolutional layer. This single convolutional layer provides useful translational invariance. The resulting descriptor is then fed through multiple layers of recursive neural networks. These recursive layers with various weights allow the model to learn compositional features and parts of the object. Finally, the results from both the RGB stream and depth stream are combined and merged through a SoftMax classifier to obtain the classification of the 3D object.

One of the benefits of this approach is that the input is simple. RGB-D data is not computationally intensive compared to other inputs such as volumetric voxels or point clouds. This makes RGB-D approaches fast and efficient.

In terms of accuracy, when performed on the RGB-D dataset of Lai et al. in [8], the CNN-RNN developed by Socher et al. in [7] outperformed traditional machine learning methods that were modified to perform on 3D RGB-D datasets.

<p align="center">
  <img src="/assets/img/Table3.PNG" />
</p>

## CNN with Transfer Learning
To build off this, Alexandre in [9] modified CNN-RNN to enable transfer learning between input channels. To do this, four independent convolutional neural networks are utilized, one for each channel (three color channels and one depth channel), instead of a single CNN. After training each CNN for every channel, the results are then combined to produce a final decision. Transfer learning can be applied between the weights of the CNNs to obtain better error and time results.

Deep learning can be effectively applied to RGB-D data for object classification. Furthermore, separately running each input channel on its own CNN provides more accurate results. Furthermore, transfer learning can also be applied to save computational time during the training process. However, RGB-D data has its limitations. For one, RGB-D methods do not learn the full geometry of a 3D object. Instead, they can only infer about the objects shape with the information from the additional depth channel. Therefore, multi-view representations and volumetric data provide better insights into the full geometry of a 3D object.

# Point Clouds 
Point clouds are sets of geometric data points in space. A single point cloud consists of an X, Y, and Z coordinate. If there is color, the point cloud saves the RGB intensity and becomes four dimensional. 

To generate point clouds, a 3D scanner is needed to scan an object. Through a lengthy iterative process, the scanner subsequently generates point cloud data of the specific object. The scanner captures each point clouds specific locational coordinate as well as its color intensity. 

Unlike voxels, multi-view image renderings, or RGB-D data, point clouds are non-Euclidean meaning they are not in a regular format and lack the ability to apply deep learning architectures. For this reason, most researchers typically transform point cloud data to regular 3D voxels or multi-view images. However, transforming point clouds can impose limitations like increasing the amount of memory required and introducing quantization artifacts that causes invariances in our data. These limitations make adapting deep learning architectures of accept raw point cloud data an important research topic.

## PointNet
One of the first deep learning architectures to accept raw 3D point cloud data as an input was PointNet proposed by Qi et al. in [10]. PointNet consists of three main modules: a symmetric function layer, a combination structure, and a join alignment network. The symmetric function processes all the data into one canonical form and learns the key points of the point cloud data. This information is fed into the RNN module which learns the point cloud. This RNN module is invariant to the sequence of the input order of point clouds. Finally, the join alignment network aggregates all the resulting features using the max-pooling operation. 

PointNet’s primary benefit is that it can exploit the geometry of point clouds directly unlike traditional point cloud methods. PointNet also proved robust to missing data. Qi et al. conducted a robustness test on PointNet and VoxNet by randomly dropping out input points by a certain ratio prior to training. VoxNet’s accuracy dramatically dropped when half of the input points were missing, from 86.3% to 46% accuracy. On the other hand, PointNet’s accuracy only dropped by 3.7%. The reason PointNet is robust to missing data is that it utilizes critical points to summarize the shape and therefore does not need every point in the dataset. 

Another benefit to PointNet is that it has low complexity. While other architectures like MVCNNs offer higher classification accuracy, PointNet is more efficient in terms of computational cost. Qi et al. measured the computational performance of PointNet and found that it was able to process more than one million points per second for classification tasks. This gives PointNet the potential to be used for real-time applications. 

Despite PointNet’s efficient time complexity and robustness to missing data, is it not able to fully exploit the object’s local structure. Exploiting an object’s structure is important for convolutional architecture accuracy because CNNs take in data defined on regular grids as an input and are able to progressively capture features at increasingly larger scales. Without local structure, PointNet’s accuracy suffers.

## PointNet++
Building upon the foundations of PointNet, Qi et al. proposed PointNet++ in [11]. PointNet++ aims to learn local features by exploiting metric space distances in a hierarchical fashion. PointNet++ works by first partitioning the set of points into overlapping regions by a distance metric. Local features are then extracted to capture the geometric structures. This extraction process is repeated until every feature of the whole point set is obtained.

Qi et al. tested PointNet++ on the ModelNet40 dataset and were able to achieve a higher classification accuracy than the original PointNet architecture and MVCNN. These results prove that PointNet++ can learn features efficiently and accurately. 

One of the drawbacks of PointNet++ is its complexity. PointNet++’s architecture is complex which increases the size of features and the computational intensity required to obtain these features.

In all, point clouds show promising results in the field of object classification. Architectures like PointNet and PointNet++ allow researchers to derive high level features directly from point cloud inputs. Despite their high accuracy in object classification tasks, point cloud methods suffer in terms of computational time due to the unstructured nature of point clouds representations. 

## Kd-Networks
A notable deep learning architecture that does not operate on point clouds directly is Kd-Networks. Proposed by Klovkov et al. in [12], Kd-Networks utilize a common indexing structure called a kd-tree to perform computations, share learnable parameters, and compute hierarchical representations.

To apply deep learning to CNNs in three dimensions, academics traditionally voxelized 3D models onto uniform voxel grids. The process of voxelization usually resulted in large memory requirements and large computational intensity. Kd-Networks solve this problem by mimicking the architecture of CNNs with kd-trees. 

Klovkov et al. measured the accuracy of Kd-Networks with various data augmentations on the ModelNet10 and ModelNet40 datasets for 3D object classification. The data augmentations that were measured were deterministic kd-trees, randomized kd-trees, translation augmentation, and anisotropic scaling augmentation. During training either a deterministic or randomized kd-tree is constructed and used to perform the forward-backward pass in the Kd-Network. Data augmentations are then applied during both training and testing and the classification accuracy of each is recorded. Translation augmentations (TR) are proportional translations along every axis. Anisotropic scaling augmentations (AS) involve rescaling over two horizontal axes by a number sampled from the range of 0.66 to 1.5.

<p align="center">
  <img src="/assets/img/Table4.PNG" />
</p>

Based off Klovkov’s benchmarks, Kd-Networks were able to accurately classify 3D objects in both the ModelNet10 and ModelNet40 datasets. Although Kd-Network’s classification accuracy falls short of MVCNNs, data augmentations applied to kd-trees allowed the algorithm to outperform PointNet. Compared to the other top-performing convolutional architectures, Kd-Networks are efficient and therefore have low computational intensity.

Due to the efficiency and accuracy of Kd-Networks, Klovkov suggests other hierarchical 3D space partitions like octrees, PCA-trees and bounding volume hierarchies could also be investigated and researched for deep learning architectures.

# Conclusion
As the demand for new technologies such as autonomous driving, virtual reality, and 3D medical image processing develops, the need for quality deep learning architectures on 3D data representations will continue to grow. This survey showed the numerous deep learning architectures on multi-view, volumetric, RGB-D, and point cloud data representations. 

Multi-view data representations provide accurate solutions to model and classify 3D objects. They model 3D objects through a series of views that are individually ran through pretrained CNNs. The results of each view are then collated and fed through a final CNN for a classification decision. Multi-view representations present a challenge with the optimized number of views one should use within the architecture. If the number of views is too low, MVCNNs will not model the full geometry of the object. However, if the number of views is too high, it could lead to unnecessary computational intensity and an overfitted model. 

Volumetric representations utilized voxels to represent 3D objects through a process of conversion called voxelization. While volumetric approaches like ShapeNet, VoxNet, and LightNet provide simple yet effective deep learning architectures, they suffer against more accurate multi-view methods and from computational intensity. 

RGB-D image renderings are developed through a specialized RGB-D camera like Microsoft’s Kinect. To classify an object, its RGB and depth image are passed through two parallel CNNs. The resulting features are collated by a SoftMax classifier to obtain the classification result. Despite having an efficient runtime with the use of transfer learning, deep learning techniques on RGB-D image renderings do not learn the full geometry of the object. However, RGB-D sensors and cameras are easy to obtain and simple to use which contributes to the surge in RGB-D data and research.

Unlike the previous 3D data representations, point clouds pose an interesting topic due to their lack of structure. Traditionally, point clouds were transformed to regular 3D voxels to apply deep learning techniques. However, recent architectures such as PointNet and PointNet++ allow the direct input of point clouds into their deep learning architecture. Moreover, alternative hierarchical partitioning structures like Kd-Networks were developed to mimic CNNs with the use of kd-trees. Kd-Networks proved accurate and efficient when compared against other deep learning architectures. While the PointNet architecture family can classify 3D objects to a high degree of accuracy, they suffer in terms of computational intensity due to their unstructured representation. Despite their limitations, point clouds are a trending topic in research due to their potential to be used in a wide variety of applications from virtual reality to autonomous driving.

[1]	Hang Su, Subhransu Maji, E. Kalogerakis, & E. Learned-Miller (2015). Multi-view Convolutional Neural Networks for 3D Shape Recognition2015 IEEE International Conference on Computer Vision (ICCV), 945-953.

[2]	Jong-Chyi Su, Matheus Gadelha, R. Wang, & Subhransu Maji (2018). A Deeper Look at 3D Shape ClassifiersArXiv, abs/1809.02560.

[3]	Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang, and Jianxiong Xiao. 2015. 3D ShapeNets: A Deep Representation for Volumetric Shapes. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE

[4]	Daniel Maturana and Sebastian Scherer. 2015. Voxnet: A 3d convolutional neural network for real-time object recognition. In Intelligent Robots and Systems (IROS), 2015 IEEE/RSJ International Conference on. IEEE, 922–928.

[5]	Maxim Tatarchenko, Alexey Dosovitskiy, and Thomas Brox. 2017. Octree generating networks: Efficient convolutional architectures for high-resolution 3d outputs. CoRR, abs/1703.09438 (2017).

[6]	Shuaifeng Zhi, Yongxiang Liu, Xiang Li, and Yulan Guo. 2018. Toward real-time 3D object recognition: A lightweight volumetric CNN framework using multitask learning. Computers & Graphics 71 (2018), 199 – 207.

[7]	Richard Socher, Brody Huval, Bharath Bath, Christopher D Manning, and Andrew Y Ng. 2012. Convolutional-recursive deep learning for 3d object classification. In Advances in Neural Information Processing Systems. NIPS, 656–664.

[8]	K. Lai, L. Bo, X. Ren, and D. Fox. A Large-Scale Hierarchical Multi-View RGB-D Object Dataset. In ICRA, 2011

[9]	Luís A Alexandre. 2016. 3D object recognition using convolutional neural networks with transfer learning between input channels. In Intelligent Autonomous Systems 13. Springer, 889–898.

[10]	Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. 2017. Pointnet: Deep learning on point sets for 3d classification and segmentation. Proc. Computer Vision and Pattern Recognition (CVPR), IEEE 1, 2 (2017), 4.

[11]	Charles R Qi, Li Yi, Hao Su, and Leonidas J Guibas. 2017. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in Neural Information Processing Systems. NIPS, 5105–5114

[12]	Roman Klokov and Victor Lempitsky. 2017. Escape from cells: Deep kd-networks for the recognition of 3d point cloud models. In 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, 863–872.

[13]	Yassine Ghouzam. 2017. Intorduction to CNN Keras – 0.997 (top 6%). Retrieved December 14, 2020 from https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

[14]	Deep Learning Advances on Different 3D Data Representations: A Survey - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Two-stream-CNNs-for-3D-object-recognition-on-RGB-D-data-80_fig3_326870805 [accessed 14 Dec, 2020]