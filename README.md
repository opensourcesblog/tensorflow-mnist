# Tensorflow MNIST
How to preprocess for MNIST? I didn't find any good resources for it so I decided to reasearch and try it out and 
explain it simple with steps in Python. You can find the result on my blog [OpenSourcES](http://opensourc.es/blog/tensorflow-mnist).

## Usage

This code should be working on Python 3.6 if opencv and tensorflow and a few other packages are installed.
See `requirements.txt`

You should be able to run `python mnist.py` and `python predict_interface_usage.py test_2` where `test_2` is the filename (without extension) of an image in img/ 

### mnist.py
 `SUCCESS` will be in the form of something like this:
 > 0.9145
 >
 > [8 0 4 3]
 >
 > 1.0
 
### predict_interface_usage.py
 A photograph of handwritten digits as input (`img/`),
 `SUCCESS` will write an output with predictions to the command
 prompt and it will generate an image with the predictions in `pro-img/`

