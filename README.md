# Importance-Weighted-Autoencoders
A PyTorch implementation of both IWAE and VAE experiments in the paper *Importance Weighted Autoencoders* 
by Yuri Bruda, Roger Grosse and Ruslan Slakhutdinov. 
https://arxiv.org/pdf/1509.00519.pdf

## Usage

```
usage: main.py [-h] [--model {iwae,vae}] [--layer {1,2}] [--k K]
               [--epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
               [--device DEVICE] [--data_dir DATA_DIR] [--save_dir SAVE_DIR]

Demo for Training VAE

optional arguments:
  -h, --help            show this help message and exit
  --model {iwae,vae}    The model, use IWAE or VAE
  --layer {1,2}         The number of stochastic layer(s) used in IWAE or VAE
  --k K                 Choose the k in the k-sample IWAE or VAE
  --epochs NUM_EPOCHS   Total number of epochs
  --batch_size BATCH_SIZE
                        The batch size
  --device DEVICE       Index of device
  --data_dir DATA_DIR   The directory of your dataset
  --save_dir SAVE_DIR   The directory to save your trained model
```

It can be seen by running
```
python main.py --help
```

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


