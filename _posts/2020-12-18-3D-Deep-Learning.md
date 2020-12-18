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
According to Su et al. in [1], building deep learning classifiers for 3D models upon 2D image renderings dramatically outperformed classifiers built directly on 3D representations. One reason for the outperformance is that 3D representations rely on a significant reduction in resolution. On the other hand, 2D image renderings can be trained with deep learning quickly without the need for changes in resolution. Su et al. applied a multi-view convolutional neural network on ModelNet40, a comprehensive dataset of 3D CAD models of objects, and found that their model outperformed traditional 3D models. Their multi-view convolutional neural network 




Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
