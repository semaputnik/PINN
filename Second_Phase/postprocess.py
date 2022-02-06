import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import sklearn.metrics as m


if __name__ == "__main__":
    path_for_run = "runs/1642966313"

    predicted_data_path = f"{path_for_run}/predicted_data.csv"
    source_data_path = f"{path_for_run}/source_data.csv"

    predicted_data: DataFrame = pd.read_csv(predicted_data_path)

    source_data: DataFrame = pd.read_csv(source_data_path)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    ax.plot(source_data["t"], source_data["E"], 'violet', alpha=0.5, lw=2, label='Exposed')
    ax.plot(source_data["t"], predicted_data["E"], 'dodgerblue', alpha=0.9, lw=2, label='Exposed Prediction', linestyle='dashed')

    ax.plot(source_data["t"], source_data["I"], 'darkgreen', alpha=0.5, lw=2, label='Infected')
    ax.plot(source_data["t"], predicted_data["I"], 'green', alpha=0.9, lw=2, label='Infected Prediction', linestyle='dashed')

    ax.plot(source_data["t"], source_data["R"], 'blue', alpha=0.5, lw=2, label='Recovered')
    ax.plot(source_data["t"], predicted_data["R"], 'teal', alpha=0.9, lw=2, label='Recovered Prediction', linestyle='dashed')

    ax.set_xlabel('Time /days', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig(path_for_run + "/train_result.png")

    print('MSE(S): ', round(m.mean_squared_error(source_data["S"], predicted_data["S"]), 4))
    print('MSE(E): ', round(m.mean_squared_error(source_data["E"], predicted_data["E"]), 4))
    print('MSE(I): ', round(m.mean_squared_error(source_data["I"], predicted_data["I"]), 4))
    print('MSE(R): ', round(m.mean_squared_error(source_data["R"], predicted_data["R"]), 4))

    print('R^2(S): ', round(m.r2_score(source_data["S"], predicted_data["S"]), 4))
    print('R^2(E): ', round(m.r2_score(source_data["E"], predicted_data["E"]), 4))
    print('R^2(I): ', round(m.r2_score(source_data["I"], predicted_data["I"]), 4))
    print('R^2(R): ', round(m.r2_score(source_data["R"], predicted_data["R"]), 4))

