# This is the python script to train your IWAE or VAE model

import torch
import torch.nn.functional as F
import argparse
import numpy as np
from model import VAE_1
from model import VAE_2
from utils.test import test_function_1
from utils.test import test_function_2
from tqdm import tqdm
import time


def load_binary_mnist(d_dir):
    train_file = d_dir + '/BinaryMNIST/binarized_mnist_train.amat'
    valid_file = d_dir + '/BinaryMNIST/binarized_mnist_valid.amat'
    test_file = d_dir + '/BinaryMNIST/binarized_mnist_test.amat'
    mnist_train = np.concatenate([np.loadtxt(train_file), np.loadtxt(valid_file)])
    mnist_test = np.loadtxt(test_file)
    return mnist_train, mnist_test


def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        param_group['lr'] = lr / 10 ** (1 / 7)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training VAE")
    parser.add_argument("--model", dest='model', default='iwae',
                        choices=('iwae', 'vae'),
                        help="The model, use IWAE or VAE")
    parser.add_argument("--layer", dest='layer', default=1, type=int,
                        choices=(1, 2),
                        help="The number of stochastic layer(s) used in IWAE or VAE")
    parser.add_argument("--k", dest="k", default=5, type=int,
                        help="Choose the k in the k-sample IWAE or VAE ")
    parser.add_argument("--epochs", dest='num_epochs', default=102, type=int,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=20, type=int,
                        help="The batch size")
    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of device')
    parser.add_argument("--data_dir", dest='data_dir', default="../../dataset",
                        help="The directory of your dataset")
    parser.add_argument("--save_dir", dest='save_dir', default="./saved_models",
                        help="The directory to save your trained model")
    args = parser.parse_args()

    # load the data
    training_set, test_set = load_binary_mnist(args.data_dir)
    print('training set shape', training_set.shape)
    print('test set shape', test_set.shape)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)

    # define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    k = args.k
    batch_size = args.batch_size
    if args.model == 'iwae':
        # this index_vec will be used in the training of iwae
        index_vec = torch.tensor([i * k for i in range(batch_size)]).to(device)

    if args.layer == 1:
        net = VAE_1(args).to(device)
    else:
        net = VAE_2(args).to(device)

    # train the model
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate
                                 , betas=(0.9, 0.999), eps=1e-04)
    start = time.time()

    for epoch in range(args.num_epochs):
        # change learning rate
        if epoch in [1, 4, 13, 40, 121, 364, 1093, 3280]:
            adjust_learning_rate(optimizer)
            print("learning rate decay")

        # perform a test
        if epoch % 10 == 1:
            if args.layer == 1:
                nll, A_u = test_function_1(net, test_set, device)
            else:
                nll, A_u = test_function_2(net, test_set, device)
            print('NLL:', nll, 'active units:', A_u)

        # training process
        running_loss = 0.0
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for i, data in enumerate(train_loader, 0):
                # forward pass
                train = data.view(-1, 1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = net(train.float())

                # calculate the loss
                log_prob_condi = torch.sum(output[0] * train - torch.log(1 + torch.exp(output[0])), 2)

                if args.layer == 1:
                    # stochastic layer
                    h = (output[1] + torch.exp(output[2]) * output[3])

                    # log weights, unnormalized
                    log_weights = log_prob_condi - (h * h).sum(2) / 2 + (output[3] * output[3]).sum(2) / 2 + output[
                        2].sum(2)
                else:

                    # stochastic layer h1 minus mu3
                    h1 = output[1] + torch.exp(output[4]) * output[7] - output[3]

                    # stochastic layer h2
                    h2 = (output[2] + torch.exp(output[5]) * output[8])

                    # log weights, unnormalized
                    log_p_h1_h2 = -(h1 * h1 / torch.exp(2 * output[6])).sum(2) / 2 - output[6].sum(2)
                    log_q_h1_x = -(output[7] * output[7]).sum(2) / 2 - output[4].sum(2)
                    log_q_h2_h1 = -(output[8] * output[8]).sum(2) / 2 - output[5].sum(2)
                    log_weights = log_prob_condi + log_p_h1_h2 - (h2 * h2).sum(2) / 2 - log_q_h1_x - log_q_h2_h1

                if args.model == 'vae':
                    loss = -log_weights.mean()
                else:
                    # sample one index from 1,2,...,k accoring to the normalized weights
                    temp = torch.exp(F.log_softmax(log_weights - log_weights.min(1)[0].view(-1, 1), 1))
                    temp1 = torch.multinomial(temp, 1).flatten() + index_vec

                    # estimate loss
                    loss = -torch.take(log_weights, temp1).mean()

                # back propagation and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # progress bar
                progress_bar.update(data.size(0))

        # print out the average of running loss
        print('[%d] loss: %.3f' % (epoch + 1, running_loss*batch_size/60000 ))

    print('Finished Training. Total time cost:' + str(time.time() - start))

    # save the model
    PATH = args.save_dir + '/' + args.model + '_layer_' + str(args.layer) + '_k_' + str(k) + '_' + '.pth'
    torch.save(net.state_dict(), PATH)
