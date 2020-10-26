# Importance-Weighted-Autoencoders
A PyTorch implementation of both IWAE and VAE experiments in the paper *Importance Weighted Autoencoders* 
by Yuri Bruda, Roger Grosse and Ruslan Slakhutdinov. 
https://arxiv.org/pdf/1509.00519.pdf

## Decoder Type
Bernoulli decoder

## Batch Size
The default value is 20. Choose a larger batch size may help the code run faster, but will likely to get a worse model, 
since its training is quite sensitive to batch size.

## Device
As instructed in the paper, this implementation uses $L_{5000}$ to estimate $log(p_{\theta}(x))$,
so it will be much faster if run the code on GPU instead of CPU. Or, one can change the *num_repeat*
in the *test.py* to a smaller integer instead of 5000.

## Datasets
The binarized version of the MNIST dataset can be downloaded from
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat


