import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd

def ode(y, t, alpha, beta, gamma):
    S, E, I, R = y
    dSdt = - alpha * S * I
    dEdt = alpha * S * I - beta * E
    dIdt = beta * E - gamma * I
    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt


def draw(source_data, predicted_data, add_predicted=True):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    # ax.plot(source_data.index, source_data["S"], 'pink', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(source_data.index, predicted_data["S"], 'red', alpha=0.9, lw=2, label='Susceptible Prediction', linestyle='dashed')

    #ax.plot(source_data.index[:115], source_data["E"][:115], 'red', alpha=0.5, lw=2, label='Exposed')
    #ax.plot(source_data.index[114:], source_data["E"][114:], 'gold', alpha=0.5, lw=2, label='Future Exposed')


    ax.plot(source_data.index, source_data["I"], 'darkgreen', alpha=0.5, lw=2, label='Infected')


    # ax.plot(source_data.index[:115], source_data["R"][:115], 'darkblue', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(source_data.index[114:], source_data["R"][114:], 'gold', alpha=0.5, lw=2, label='Future Recovered')


    if (add_predicted):
        # ax.plot(source_data.index, predicted_data["E"], 'coral', alpha=0.9, lw=2,
                #label='Exposed Prediction', linestyle='dashed')
        ax.plot(source_data.index, predicted_data["I"]*N, 'green', alpha=0.9, lw=2,
                label='Infected Prediction', linestyle='dashed')
        #ax.plot(source_data.index, predicted_data["R"], 'blue', alpha=0.9, lw=2,
                #label='Recovered Prediction', linestyle='dashed')


    ax.set_xlabel('Day', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.show()


N = 5000000


E0 = 1/N
I0 = 0.0
R0 = 0.0
S0 = 1 - E0 - I0 - R0


gamma = 1/0.5814
alpha = 1.3809/gamma
beta = 1/0.1149



#(alpha, beta, gamma) = (1.6275, 49.7056, 1.7488)

start_day = 1


if __name__ == "__main__":
    print(alpha, beta, gamma)

    source_data = pd.read_csv("data/covid_data.csv")[1:]

    print(source_data[["I","R","E","t"]].values[:,2])

    # A grid of time points (in days)
    t = np.linspace(0, len(source_data), len(source_data))

    # Initial conditions
    #y0 = N - source_data["E"].iloc[start_day], source_data["E"].iloc[start_day], source_data["I"].iloc[start_day], source_data["R"].iloc[start_day]
    y0 = S0, E0, I0, R0
    print(y0)

    ret = odeint(ode, y0, t, args=(alpha, beta, gamma))
    data = pd.DataFrame(ret, columns=["S", "E", "I", "R"])
    data["t"] = t

    print(data["I"])


    draw(source_data, data)
