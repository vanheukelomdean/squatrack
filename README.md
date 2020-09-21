# squatrack
This project is an application of convolutional pose machine used for human pose estimation to provide analysis of power lifting form. 

## Design Plans 

Proof of concept will first be prototyped using the tensorflow/opencv python api, trained on the mpii datatset, then the trained model will be packaged and deployed for mobile.

The current solution is an implementation of this [2016 research paper](https://arxiv.org/abs/1602.00134), however the implementation may be reconsidered using motion tracking for improved frame rate
