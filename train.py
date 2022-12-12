# training loop

from agent import Agent
from game import Game
import numpy as np
from plot import plot, plot_loss, save

VERSION = "max_obstacle"

NUM_GAMES = 200

if __name__ == "__main__":
    game = Game(train=True)
    agent = Agent()
    scores = []
    mean_scores = []
    losses = []

    for i in range(NUM_GAMES):
        loss = []

        game_over = False
        observation = np.array(game.grayscale_image(), dtype=np.float32)

        while not game_over:
            game.draw()
            action = agent.choose_action(observation)
            game.move(action)
            game_over, reward = game.update()

            # only try to learn when the player can act
            if game.player_y == 0:
                observation_next = np.array(game.grayscale_image(), dtype=np.float32)

                agent.store_transition(observation, action, reward,
                                    observation_next, game_over)
                curr_loss = agent.learn()
                if curr_loss:
                    loss.append(curr_loss)
            
                observation = observation_next

        scores.append(game.score)
        avg_score = np.mean(scores[-25:])
        mean_scores.append(avg_score)

        loss = np.mean(loss)
        losses.append(loss)

        # display training info
        print(f"Game {i}, Epsilon: {agent.epsilon: .3f} - Average Score (last 25): {avg_score: .3f}")
        print(f"Final Score: {game.score: .3f}, Loss: {loss: .3f}")
        # plot(scores, mean_scores)
        plot_loss(losses)

        game.reset()
        agent.decay_epsilon()

    agent.model.save(file_name=f"model_{VERSION}.pth")
    save("img/{VERSION}_loss_plot.png")
