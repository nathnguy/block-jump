import torch
from game import Game
import numpy as np
from model import BlockNet

NUM_GAMES = 500

 # get average score for random games 
def baseline_test():
    game = Game(train=True)
    scores = []
    for _ in range(NUM_GAMES):
        game_over = False
        while not game_over:
            game.draw()
            action = np.random.choice(2, 1).item()
            game.move(action)
            game_over, _ = game.update()

        scores.append(game.score)
        game.reset()

    avg_score = np.mean(scores)
    print(f"average: {avg_score}")

def model_test(model_name):
    model = BlockNet()
    model.load_state_dict(torch.load(f"./model/{model_name}"))
    model.eval()
    game = Game(train=True)
    scores = []

    # TODO run games
    with torch.no_grad():
        for i in range(NUM_GAMES):
            game_over = False
            while not game_over:
                game.draw()

                # get predicted action
                image = np.array([game.grayscale_image()], dtype=np.float32)
                state = torch.tensor(image).to(model.device)
                actions = model.forward(state)
                action = torch.argmax(actions).item()

                game.move(action)
                game_over, _ = game.update()

            scores.append(game.score)
            avg_score = np.mean(scores)
            print(f"{i + 1} Games, Average Score: {avg_score}")

            game.reset()


if __name__ == "__main__":
    model_test("model.pth")
