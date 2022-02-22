import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def draw(df):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    ax.plot(df.index, df["E"], 'darkgreen', alpha=0.5, lw=2, label='Exposed', linestyle='dashed')
    ax.plot(df.index, df["I"], 'blue', alpha=0.5, lw=2, label='Infected', linestyle='dashed')
    ax.plot(df.index, df["R"], 'red', alpha=0.5, lw=2, label='Recovered', linestyle='dashed')

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


N = 5e6


if __name__ == "__main__":
    data = pd.read_csv('data/spb.combined.daily.txt', '\t', index_col=['TIME'], parse_dates=['TIME'], encoding='cp1251')[2:]
    data.drop(['HOSPITALIZED.today', 'DEATHS', 'Yandex.ACTIVITY.points', 'v1.CS', 'v2.CS', 'CONFIRMED.spb', 'PCR.tested'], axis=1, inplace=True)
    data.fillna(0, inplace=True)

    df = pd.DataFrame()
    df["E"] = data["CONFIRMED"]
    df["I"] = data["ACTIVE"]
    df["R"] = data["RECOVERED"]
    df["S"] = N - df["I"] - df["E"] - df["R"]
    df["t"] = np.arange(0.0, len(df), 1)

    draw(df)

    # save to csv file
    save(df[:120], "data/covid_data.csv")
