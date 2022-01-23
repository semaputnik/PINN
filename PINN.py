import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pandas as pd


class PINN(nn.Module):
    def __init__(self, t, S_data, E_data, I_data, R_data):
        super(PINN, self).__init__()

        # population size
        self.N = 59e6

        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t), 1))

        # for the compartments we just need to convert them into tensors
        self.S = torch.tensor(S_data)
        self.E = torch.tensor(E_data)
        self.I = torch.tensor(I_data)
        self.R = torch.tensor(R_data)

        self.losses = []

        # setting the parameters
        self.alpha_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.beta_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))
        self.gamma_tilda = torch.nn.Parameter(torch.rand(1, requires_grad=True))

        # find values for normalization
        self.S_max = max(self.S)
        self.E_max = max(self.E)
        self.I_max = max(self.I)
        self.R_max = max(self.R)
        self.S_min = min(self.S)
        self.E_min = min(self.E)
        self.I_min = min(self.I)
        self.R_min = min(self.R)

        # normalize
        self.S_hat = (self.S - self.S_min) / (self.S_max - self.S_min)
        self.E_hat = (self.E - self.E_min) / (self.E_max - self.E_min)
        self.I_hat = (self.I - self.I_min) / (self.I_max - self.I_min)
        self.R_hat = (self.R - self.R_min) / (self.R_max - self.R_min)

        # matrices (x4 for S,E,I,R) for the gradients
        self.m1 = torch.zeros((len(self.t), 4))
        self.m1[:, 0] = 1
        self.m2 = torch.zeros((len(self.t), 4))
        self.m2[:, 1] = 1
        self.m3 = torch.zeros((len(self.t), 4))
        self.m3[:, 2] = 1
        self.m4 = torch.zeros((len(self.t), 4))
        self.m4[:, 3] = 1

        # NN
        self.net_seir = self.Net_seir()
        self.params = list(self.net_seir.parameters())
        self.params.extend(list([self.alpha_tilda, self.beta_tilda, self.gamma_tilda]))

    @property
    def alpha(self):
        return torch.tanh(self.alpha_tilda)

    @property
    def beta(self):
        return torch.tanh(self.beta_tilda)

    @property
    def gamma(self):
        return torch.tanh(self.gamma_tilda)

    class Net_seir(nn.Module):
        def __init__(self):
            super(PINN.Net_seir, self).__init__()

            self.fc1 = nn.Linear(1, 20)
            self.fc2 = nn.Linear(20, 20)
            self.fc3 = nn.Linear(20, 20)
            self.fc4 = nn.Linear(20, 20)
            self.fc5 = nn.Linear(20, 20)
            self.fc6 = nn.Linear(20, 20)
            self.fc7 = nn.Linear(20, 20)
            self.fc8 = nn.Linear(20, 20)
            self.out = nn.Linear(20, 4)

        def forward(self, t_batch):
            seir = F.relu(self.fc1(t_batch))
            seir = F.relu(self.fc2(seir))
            seir = F.relu(self.fc3(seir))
            seir = F.relu(self.fc4(seir))
            seir = F.relu(self.fc5(seir))
            seir = F.relu(self.fc6(seir))
            seir = F.relu(self.fc7(seir))
            seir = F.relu(self.fc8(seir))
            seir = self.out(seir)
            return seir

    def net_f(self, t_batch):

        # pass the timesteps batch to the neural network
        seir_hat = self.net_seir(t_batch)

        S_hat, E_hat, I_hat, R_hat = seir_hat[:, 0], seir_hat[:, 1], seir_hat[:, 2], seir_hat[:, 3]

        # S_t
        seir_hat.backward(self.m1, retain_graph=True)
        S_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # E_t
        seir_hat.backward(self.m2, retain_graph=True)
        E_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # I_t
        seir_hat.backward(self.m3, retain_graph=True)
        I_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # R_t
        seir_hat.backward(self.m4, retain_graph=True)
        R_hat_t = self.t.grad.clone()
        self.t.grad.zero_()

        # unnormalize
        S = self.S_min + (self.S_max - self.S_min) * S_hat
        E = self.E_min + (self.E_max - self.E_min) * E_hat
        I = self.I_min + (self.I_max - self.I_min) * I_hat
        R = self.R_min + (self.R_max - self.R_min) * R_hat

        f1_hat = S_hat_t - (-(self.alpha / self.N) * S * I) / (self.S_max - self.S_min)
        f2_hat = E_hat_t - (((self.alpha / self.N) * S * I) - (self.beta * E)) / (self.E_max - self.E_min)
        f3_hat = I_hat_t - (self.beta * E - self.gamma * I) / (self.I_max - self.I_min)
        f4_hat = R_hat_t - (self.gamma * I) / (self.R_max - self.R_min)

        return f1_hat, f2_hat, f3_hat, f4_hat, S_hat, E_hat, I_hat, R_hat

    def train(self, n_epochs):
        # train
        logging.info('Starting training...')

        for epoch in tqdm(range(n_epochs)):
            S_pred_list = []
            E_pred_list = []
            I_pred_list = []
            R_pred_list = []

            f1, f2, f3, f4, S_pred, E_pred, I_pred, R_pred = self.net_f(self.t_batch)

            self.optimizer.zero_grad()  # zero grad

            # append the values to plot later (unnormalized)
            S_pred_list.append(self.S_min + (self.S_max - self.S_min) * S_pred)
            E_pred_list.append(self.E_min + (self.E_max - self.E_min) * E_pred)
            I_pred_list.append(self.I_min + (self.I_max - self.I_min) * I_pred)
            R_pred_list.append(self.R_min + (self.R_max - self.R_min) * R_pred)

            # calculate the loss
            loss = (torch.mean(torch.square(self.S_hat - S_pred)) +
                    torch.mean(torch.square(self.E_hat - E_pred)) +
                    torch.mean(torch.square(self.I_hat - I_pred)) +
                    torch.mean(torch.square(self.R_hat - R_pred)) +
                    torch.mean(torch.square(f1)) +
                    torch.mean(torch.square(f2)) +
                    torch.mean(torch.square(f3)) +
                    torch.mean(torch.square(f4))
                    )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())

            if epoch % 100 == 0:
                logging.info(
                    f'Epoch {epoch}: (alpha, beta, gamma) = '
                    + f'({round(self.alpha.item(), 4)}, {round(self.beta.item(), 4)}, {round(self.gamma.item(),4)})')

        return pd.DataFrame({"S": S_pred_list[0].detach().numpy(), "E": E_pred_list[0].detach().numpy(),
                             "I": I_pred_list[0].detach().numpy(), "R": R_pred_list[0].detach().numpy()})
