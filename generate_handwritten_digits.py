# After running main.py to train the model, this script can
# generate handwritten digits using one of the saved model


import torch
from model import VAE_1
from model import VAE_2
import argparse
import matplotlib.pyplot as plt


def generate_1(net, z):
    x = torch.tanh(net.fc4(z))
    x = torch.tanh(net.fc5(x))
    out = torch.sigmoid(net.fc6(x))
    return out.detach().view(28, 28)


def generate_2(net, z):
    x = torch.tanh(net.fc7(z))
    x = torch.tanh(net.fc8(x))
    mu3 = net.fc9_mu(x)
    log_sigma3 = net.fc9_sigma(x)
    eps3 = torch.randn_like(mu3)
    h1 = mu3 + torch.exp(log_sigma3) * eps3
    x = torch.tanh(net.fc10(h1))
    x = torch.tanh(net.fc11(x))
    out = torch.sigmoid(net.fc12(x))
    return out.detach().view(28, 28)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Handwritten Digits")
    parser.add_argument("--model_dir", dest='model_dir',
                        default="./saved_models",
                        help="The directory and name of your model")
    parser.add_argument("--model", dest='model', default='iwae',
                        choices=('iwae', 'vae'),
                        help="The model, use IWAE or VAE")
    parser.add_argument("--layer", dest='layer', default=1, type=int,
                        choices=(1, 2),
                        help="The number of stochastic layer(s) used in IWAE or VAE")
    parser.add_argument("--k", dest="k", default=5, type=int,
                        help="Choose the k in the k-sample IWAE or VAE ")
    parser.add_argument("--num_row", dest='n_row',
                        default=10, help="The number of rows you want")
    parser.add_argument("--num_col", dest='n_col',
                        default=10, help="The number of columns you want")
    args = parser.parse_args()

    filename = args.model_dir + '/' + args.model + '_layer_' + str(args.layer) + '_k_' + str(args.k) + '_' + '.pth'
    if args.layer == 1:
        net = VAE_1(args)
        net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    else:
        net = VAE_2(args)
        net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    n_row, n_col = args.n_row, args.n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_row, n_col))
    for i in range(args.n_row):
        for j in range(n_col):
            z1 = torch.randn(50)
            if args.layer == 1:
                img = generate_1(net, z1)
            else:
                img = generate_2(net, z1)
            np_img = img.numpy()
            axes[i, j].imshow(np_img)

    plt.savefig(args.model + '_layer_' + str(args.layer) + '_k_' + str(args.k))
