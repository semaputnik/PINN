from First_Phase.PINN import PINN
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from shutil import copyfile
import pandas as pd
import sklearn.metrics as m
from scipy.integrate import odeint
import numpy as np


def _prepare_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _draw_loss(losses, path, i, save=False):
    plt.figure(figsize=(12, 12))
    plt.plot(losses[0:], color='teal')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    if save:
        plt.savefig(f"{path}/train_loss_{i}.png")


def _draw_u_loss(losses, path, i, save=False):
    plt.figure(figsize=(12, 12))
    plt.plot(losses, color='teal')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('U Loss', fontsize=20)
    if save:
        plt.savefig(f"{path}/train_loss_u_{i}.png")


def _draw_f_loss(losses, path, i, save=False):
    plt.figure(figsize=(12, 12))
    plt.plot(losses, color='teal')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('F Loss', fontsize=20)
    if save:
        plt.savefig(f"{path}/train_loss_f_{i}.png")


def _draw_results(source_data, predicted_data, simulated_data, path, i, save=False):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    ax.scatter(source_data.index, source_data["I"], c='red', marker="*", label='Infected')
    #ax.plot(source_data.index, predicted_data["I"], 'green', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')
    ax.plot(source_data.index, simulated_data["I"]*N, 'blue', alpha=0.9, lw=2, label='Infected Simulated', linestyle='dashed')

    ax.set_xlabel('Day', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    if save:
        plt.savefig(f"{path}/train_result_{i}.png")


def _calculate_metrics(source_data, predicted_data):
    n = len(predicted_data)

    r2_score = round(m.r2_score(source_data["I"], predicted_data["I"]), 4)
    adj_r2 = 1 - ((1 - r2_score) * (n - 1) / (n - 5 - 1))

    print('R^2: ', r2_score)
    print('Adjusted R^2: ', adj_r2)


def ode(y, t, contact_rate, infective, incubation):
    S, E, I, R = y
    alpha = 1 / incubation
    gamma = 1 / infective
    beta = contact_rate * gamma

    dSdt = - beta * S * I
    dEdt = beta * S * I - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt


N = 5e6

NUM_EPOCHS = 100000

LOSS_DRAW_START = 2500


if __name__ == "__main__":

    path_for_run = f"runs/{round(datetime.now().timestamp())}"
    _prepare_dir(path_for_run)

    logging.basicConfig(filename=f'{path_for_run}/pinn.log',
                        filemode='w',
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    # Copy source data
    copyfile("data/covid_data.csv", path_for_run + "/" + "source_data.csv")

    source_data = pd.read_csv(path_for_run + "/" + "source_data.csv")

    for i in range(1):
        pinn = PINN(source_data["t"], [source_data["S"], source_data["E"], source_data["I"], source_data["R"]], N)

        learning_rate = 1e-6
        optimizer = optim.Adam(pinn.params, lr=learning_rate)
        pinn.optimizer = optimizer

        scheduler = torch.optim.lr_scheduler.CyclicLR(pinn.optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=1000,
                                                      mode="exp_range", gamma=0.85, cycle_momentum=False)

        pinn.scheduler = scheduler

        logging.info("Start training")

        predicted_data = pinn.train(NUM_EPOCHS)
        predicted_data["t"] = source_data["t"]

        logging.info("Training ends")

        logging.info(
            f"(Alpha, Beta, Gamma): {round(pinn.contact_rate.item(), 4), round(pinn.incubation_rate.item(), 4), round(pinn.infective_rate.item(), 4)}")

        _draw_loss(pinn.losses[LOSS_DRAW_START:], path_for_run, i, True)

        _draw_u_loss(pinn.lossU_history[LOSS_DRAW_START:], path_for_run, i, True)

        _draw_f_loss(pinn.lossF_history[LOSS_DRAW_START:], path_for_run, i, True)

        predicted_data_file_name = "predicted_data.csv"

        predicted_data.to_csv(path_for_run + "/" + predicted_data_file_name)

        _calculate_metrics(source_data, predicted_data)

        y0 = source_data["S"][0], source_data["E"][0], source_data["I"][0], source_data["R"][0]
        t = np.linspace(0, len(source_data), len(source_data))

        res = odeint(ode, y0, t, args=(pinn.contact_rate.item(), pinn.infective_rate.item(), pinn.incubation_rate.item()))
        simulated = pd.DataFrame(res, columns=["S", "E", "I", "R"])

        _draw_results(source_data, predicted_data, simulated, path_for_run, i, True)
