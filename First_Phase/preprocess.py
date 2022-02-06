import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd

# population size
N = 59e6

S0 = N - 3
E0 = 3
I0 = 0
R0 = 0

# A grid of time points (in days)
t = np.linspace(0, 500, 100)

# the probability of disease transmission per contact times the number of contacts per unit time
alpha = 0.28

# the rate of progression from exposure to infectious. It is the reciprocal of the incubation period
beta = 0.191

# the recovery rate. It is the inverse of the infectious period
gamma = 0.05

def ode(y, t, alpha, beta, gamma):
    S, E, I, R = y
    dSdt = - (alpha / N) * S * I
    dEdt = (alpha / N) * S * I - beta * E
    dIdt = beta * E - gamma * I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt


def draw(df):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    ax.plot(df["t"], df["S"], 'violet', alpha=0.5, lw=2, label='Susceptible', linestyle='dashed')
    ax.plot(df["t"], df["E"], 'darkgreen', alpha=0.5, lw=2, label='Exposed', linestyle='dashed')
    ax.plot(df["t"], df["I"], 'blue', alpha=0.5, lw=2, label='Infected', linestyle='dashed')
    ax.plot(df["t"], df["R"], 'red', alpha=0.5, lw=2, label='Recovered', linestyle='dashed')

    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig("data/data.png")


def save(df, path):
    try:
        os.remove(path)
    except OSError as error:
        print(error)
        print("File path can not be removed")

    df.to_csv(path)


if __name__ == "__main__":
    # Initial conditions
    y0 = S0, E0, I0, R0

    ret = odeint(ode, y0, t, args=(alpha, beta, gamma))
    data = pd.DataFrame(ret, columns=["S", "E", "I", "R"])
    data["t"] = t

    draw(data)

    # save to csv file
    save(data, "data/covid_data.csv")
