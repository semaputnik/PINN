import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import sklearn.metrics as m

if __name__ == "__main__":
    path_for_run = "runs/1644198028"

    predicted_data_path = f"{path_for_run}/predicted_data.csv"
    source_data_path = f"{path_for_run}/source_data.csv"

    predicted_data: DataFrame = pd.read_csv(predicted_data_path)

    source_data: DataFrame = pd.read_csv(source_data_path)

    r2_S = round(m.r2_score(source_data["S"], predicted_data["S"]), 4)
    r2_E = round(m.r2_score(source_data["E"], predicted_data["E"]), 4)
    r2_I = round(m.r2_score(source_data["I"], predicted_data["I"]), 4)
    r2_R = round(m.r2_score(source_data["R"], predicted_data["R"]), 4)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.set_facecolor('xkcd:white')

    # ax.plot(source_data.index, source_data["S"], 'pink', alpha=0.5, lw=2, label='Susceptible')
    # ax.plot(source_data.index, predicted_data["S"], 'red', alpha=0.9, lw=2, label='Susceptible Prediction', linestyle='dashed')

    ax.plot(source_data.index[:115], source_data["E"][:115], 'red', alpha=0.5, lw=2, label='Exposed')
    ax.plot(source_data.index[114:], source_data["E"][114:], 'gold', alpha=0.5, lw=2, label='Future Exposed')
    ax.plot(source_data.index, predicted_data["E"], 'coral', alpha=0.9, lw=2,
            label='Exposed Prediction', linestyle='dashed')

    ax.plot(source_data.index[:115], source_data["I"][:115], 'darkgreen', alpha=0.5, lw=2, label='Infected')
    ax.plot(source_data.index[114:], source_data["I"][114:], 'gold', alpha=0.5, lw=2, label='Future Infected')
    ax.plot(source_data.index, predicted_data["I"], 'green', alpha=0.9, lw=2,
            label='Infected Prediction', linestyle='dashed')

    ax.plot(source_data.index[:115], source_data["R"][:115], 'darkblue', alpha=0.5, lw=2, label='Recovered')
    ax.plot(source_data.index[114:], source_data["R"][114:], 'gold', alpha=0.5, lw=2, label='Future Recovered')
    ax.plot(source_data.index, predicted_data["R"], 'blue', alpha=0.9, lw=2,
            label='Recovered Prediction', linestyle='dashed')

    ax.set_xlabel('Day', fontsize=20)
    ax.set_ylabel('Number', fontsize=20)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='black', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)

    plt.savefig(path_for_run + "/train_result.png")

    r2_score = round(m.r2_score(source_data[["E", "I", "R"]][:115], predicted_data[["E", "I", "R"]][:115]), 4)

    print('R^2: ', r2_score)

    n = len(predicted_data)

    adj_r2 = 1 - ((1-r2_score)*(n-1)/(n-5-1))

    print('Adjusted R^2: ', adj_r2)
