# DigitsRecognition

## Goal

Goal: able to recognize any length of digits combination.(all digits come from MNIST dataset)

* the length is in [5, 20] (but we can change this as we want)
* each digit will be applied with [0.5, 1.0) resizing, and [0, 90] degree rotation
	

A sample is like this:

![sample](/3463700204839487.jpg)

which is: `3463700204839487`

## Ideas

end-to-end neural network sounds promising, but in practice, it's always hard, for its efficiency(takes too long
to optimize the parameters, too hungry for the training data), so the pipeline approach might be more practical.

### first idea: sliding window and classification

This is kind of like [YOLO](https://pjreddie.com/darknet/yolo/), and it's a pipeline

1. find the location of the interest(maybe a bounding box)
1. classify the object

so the difficult part will be step 1.


### 2nd idea: end-to-end approach

When we classify the ImageNet, we not only get 1 class but a vector of probability of classes. 
Naively if we have a threshold, we just need to output the classes whose probability is greater than 
the threshold. 

1. train a model against all single digits(and some variances, e.g resizing, rotation, etc.)
2. output the classes whose probability is greater than T(threshold)
