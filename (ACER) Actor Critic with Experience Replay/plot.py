import matplotlib.pyplot as plt
import numpy as np

def plot_epi_step(game_scores,game_steps):
    x = np.arange(len(game_scores))
    plt.plot(x,game_scores,label='score per game', c='r')
    plt.plot(x,game_steps,label='steps per game', c='b')
    plt.xlabel('episode')
    plt.ylabel('evolution')
    plt.legend()
    plt.show()