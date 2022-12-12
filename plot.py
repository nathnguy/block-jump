# plot training game and mean scores

import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    plt.clf()

    plt.title("Block Jump Training")
    plt.xlabel("Number of Games")
    plt.ylabel("Score (seconds alive)")
    
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)


def plot_loss(losses):
    plt.clf()
    
    plt.title("Training Loss")
    plt.xlabel("Number of Games")
    plt.ylabel("MSE Loss")
    
    plt.plot(losses)
    plt.ylim(ymin=0)
    

def save(filename):
    plt.savefig(filename)
