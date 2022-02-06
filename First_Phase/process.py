from First_Phase.PINN import PINN
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging
from shutil import copyfile
import pandas as pd

NUM_EPOCHS = 50000


def _prepare_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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

    pinn = PINN(source_data["t"],
                source_data["S"],
                source_data["E"],
                source_data["I"],
                source_data["R"])

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

    logging.info(f"(Alpha, Beta, Gamma): {round(pinn.alpha.item(), 4), round(pinn.beta.item(), 4), round(pinn.gamma.item(),4)}")

    fig = plt.figure(figsize=(12, 12))
    plt.plot(pinn.losses[0:], color='teal')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.savefig(path_for_run + "/train_loss.png")

    predicted_data_file_name = "predicted_data.csv"

    predicted_data.to_csv(path_for_run + "/" + predicted_data_file_name)
