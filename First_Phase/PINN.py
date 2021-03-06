import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn.metrics as m


class PINN(nn.Module):
    def __init__(self, t, SEIR_data, N):
        super(PINN, self).__init__()

        # population size
        self.N = N

        self.t = torch.tensor(t, requires_grad=True)
        self.t_float = self.t.float()
        self.t_batch = torch.reshape(self.t_float, (len(self.t), 1))

        # for the compartments we just need to convert them into tensors
        self.S = torch.tensor(SEIR_data[0])
        self.E = torch.tensor(SEIR_data[1])
        self.I = torch.tensor(SEIR_data[2])
        self.R = torch.tensor(SEIR_data[3])

        self.losses = []

        # setting the parameters
        self.contact_rate_tilda = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.incubation_rate_tilda = torch.nn.Parameter(torch.tensor(0.05, requires_grad=True))
        self.infective_rate_tilda = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

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
        self.net_u = self.Neural_net()
        self.params = list(self.net_u.parameters())
        self.params.extend(list([self.contact_rate_tilda, self.incubation_rate_tilda, self.infective_rate_tilda]))

        self.lossF_history = []
        self.lossU_history = []

    @property
    def contact_rate(self):
        return torch.tanh(self.contact_rate_tilda) * 0.5 + 1  # 0.5 - 1.5

    @property
    def incubation_rate(self):
        return torch.tanh(self.incubation_rate_tilda) * 0.25 + 0.25

    @property
    def infective_rate(self):
        return torch.tanh(self.infective_rate_tilda) * 0.5 + 0.5

    class Neural_net(nn.Module):
        def __init__(self):
            super(PINN.Neural_net, self).__init__()

            dim_hidden = 20

            self.fc1 = nn.Linear(1, dim_hidden)
            self.fc2 = nn.Linear(dim_hidden, dim_hidden)
            self.fc3 = nn.Linear(dim_hidden, dim_hidden)
            self.fc4 = nn.Linear(dim_hidden, dim_hidden)
            self.fc5 = nn.Linear(dim_hidden, dim_hidden)
            self.fc6 = nn.Linear(dim_hidden, dim_hidden)
            self.fc7 = nn.Linear(dim_hidden, dim_hidden)
            self.fc8 = nn.Linear(dim_hidden, dim_hidden)
            self.out = nn.Linear(dim_hidden, 4)

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

    def net_f(self, seir_hat):

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

        alpha = 1 / self.incubation_rate
        gamma = 1 / self.infective_rate
        beta = gamma * self.contact_rate / self.N

        f1_hat = S_hat_t - (-beta * S * I) / (self.S_max - self.S_min)
        f2_hat = E_hat_t - ((beta * S * I) - (alpha * E)) / (self.E_max - self.E_min)
        f3_hat = I_hat_t - (alpha * E - gamma * I) / (self.I_max - self.I_min)
        f4_hat = R_hat_t - (gamma * I) / (self.R_max - self.R_min)

        return [f1_hat, f2_hat, f3_hat, f4_hat]

    def lossF(self, f_results):
        loss = 0.0
        l = []

        for f in f_results:
            f_res = torch.mean(torch.square(f))
            loss += f_res
            l.append(f_res.item())

        return loss, l

    def lossU(self, seir_hat):
        S_pred, E_pred, I_pred, R_pred = seir_hat[:, 0], seir_hat[:, 1], seir_hat[:, 2], seir_hat[:, 3]
        loss = (torch.mean(torch.square(self.I_hat - I_pred)) +
                torch.mean(torch.square(self.S_hat - S_pred)) +
                torch.mean(torch.square(self.E_hat - E_pred)) +
                torch.mean(torch.square(self.R_hat - R_pred)))
        return loss

    def train(self, n_epochs):
        logging.info('Starting training...')

        for epoch in tqdm(range(n_epochs)):
            E_pred_list = []
            I_pred_list = []
            R_pred_list = []

            seir_hat = self.net_u(self.t_batch)

            f_residuals = self.net_f(seir_hat)

            S_pred, E_pred, I_pred, R_pred = seir_hat[:, 0], seir_hat[:, 1], seir_hat[:, 2], seir_hat[:, 3]

            self.optimizer.zero_grad()  # zero grad

            # append the values to plot later (unnormalized)
            E_pred_list.append(self.E_min + (self.E_max - self.E_min) * E_pred)
            I_pred_list.append(self.I_min + (self.I_max - self.I_min) * I_pred)
            R_pred_list.append(self.R_min + (self.R_max - self.R_min) * R_pred)

            lossU = self.lossU(seir_hat)
            lossF, l_ar = self.lossF(f_residuals)

            # calculate the loss
            loss = (lossU + lossF)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.lossF_history.append(lossF)
            self.lossU_history.append(lossU)
            self.losses.append(loss.item())

            if epoch % 100 == 0:
                logging.info(
                    f'Epoch {epoch}: (Contact_rate = 1.09, Incubation_rate = 0.02, Infective_rate = 0.73) = '
                    + f'({round(self.contact_rate.item(), 4)}, {round(self.incubation_rate.item(), 4)}, {round(self.infective_rate.item(), 4)})')
                logging.info(f"Loss: {loss.item()}, LossU: {lossU}, LossF: {lossF}, loss_Fs:{l_ar}")
                #logging.info(f"Loss: {loss.item()}")

        return pd.DataFrame({"E": E_pred_list[0].detach().numpy(), "I": I_pred_list[0].detach().numpy(),
                             "R": R_pred_list[0].detach().numpy()})
