# the script to calculate -log(p_{\theta}(x)) and A_u on the test set

import torch


def test_function_1(net, testset, device='cpu'):
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    nll = 0.0
    a_u = torch.zeros(50)
    num_repeat = 5000
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            test = data.view(-1, 1, 28 * 28).to(device)
            output = net(test.float())

            # stochastic layer
            eps = torch.randn_like(output[2].repeat(1, num_repeat, 1))
            h = output[1] + torch.exp(output[2]) * eps

            # output of x using the new epsilon
            output_x = torch.tanh(net.fc4(h))
            output_x = torch.tanh(net.fc5(output_x))
            output_x = net.fc6(output_x)
            log_prob_cond = torch.sum(output_x * test - torch.log(1 + torch.exp(output_x)), 2)

            # log weights, unnormalized
            log_weights = log_prob_cond - (h * h).sum(2) / 2 + (eps * eps).sum(2) / 2 + output[2].sum(2)

            # estimate log likelihood using l_5000
            l_5000 = log_weights.max(1)[0].mean() + torch.log(torch.exp(log_weights
                                                                        - log_weights.max(1)[0].view(-1, 1)).mean(
                1)).mean()
            nll -= l_5000.item()
            a_u += output[1].view(-1, 50).var(0).cpu()
    return nll/100, sum(a_u.detach().numpy()/100 > 0.01)


def test_function_2(net, testset, device='cpu'):
    testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False)
    nll = 0.0
    a_u_1 = torch.zeros(100)
    a_u_2 = torch.zeros(50)
    num_repeat = 5000
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            test = data.view(-1, 1, 28 * 28).to(device)
            output = net(test.float())

            # stochastic layer h1
            eps1 = torch.randn_like(output[1].repeat(1, num_repeat, 1))
            h1 = output[1].repeat(1, num_repeat, 1) + torch.exp(output[4].repeat(1, num_repeat, 1)) * eps1

            x = torch.tanh(net.fc4(h1))
            x = torch.tanh(net.fc5(x))

            # stochastic layer h2
            mu2 = net.fc6_mu(x)
            log_sigma2 = net.fc6_sigma(x)
            eps2 = torch.randn_like(mu2)

            x = torch.tanh(net.fc10(h1))
            x = torch.tanh(net.fc11(x))
            x = net.fc12(x)

            # log conditional prob
            log_prob_condi = torch.sum(x * test.repeat(1, num_repeat, 1), 2) - torch.sum(torch.log(1 + torch.exp(x)), 2)

            # log weights, unnormalized
            h2 = mu2 + torch.exp(log_sigma2) * eps2
            x = torch.tanh(net.fc7(h2))
            x = torch.tanh(net.fc8(x))
            mu3 = net.fc9_mu(x)
            log_sigma3 = net.fc9_sigma(x)
            h1 = h1 - mu3
            log_p_h1_h2 = -(h1 * h1 / torch.exp(2 * log_sigma3)).sum(2) / 2 - log_sigma3.sum(2)
            log_q_h1_x = -(eps1 * eps1).sum(2) / 2 - output[4].repeat(1, num_repeat, 1).sum(2)
            log_q_h2_h1 = -(eps2 * eps2).sum(2) / 2 - log_sigma2.sum(2)
            log_weights = log_prob_condi + log_p_h1_h2 - (h2 * h2).sum(2) / 2 - log_q_h1_x - log_q_h2_h1

            # estimate log likelihood using l_5000
            l_5000 = log_weights.max(1)[0].mean() + torch.log(torch.exp(log_weights
                                                                        - log_weights.max(1)[0].view(-1, 1)).mean(
                1)).mean()
            nll -= l_5000.item()
            a_u_1 += output[1].view(-1, 100).var(0).cpu()
            a_u_2 += mu2[:, 0, :].var(0).cpu()
    return nll/500, (sum(a_u_1.detach().numpy()/500 > 0.01), sum(a_u_2.detach().numpy()/500 > 0.01))




