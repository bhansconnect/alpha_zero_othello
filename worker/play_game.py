from alpha_zero_othello.config import EvaluateConfig
from alpha_zero_othello.lib import tf_util
from alpha_zero_othello.player.player import HumanPlayer, RandomPlayer
from alpha_zero_othello.player.aiplayer import AIPlayer
from alpha_zero_othello.othello import Othello
from random import random
from time import time
import glob

def start():
    config = EvaluateConfig()
    tf_util.update_memory(config.gpu_mem_fraction)

    
    models = glob.glob(config.data.model_location+"*.h5")
    if len(models) == 0:
        ai = AIPlayer(config.buffer_size, config.simulation_num_per_move)
        ai.save_weights(config.data.model_location+str(time())+".h5")
    run_games(config)
             
    
def run_games(config):
    game = Othello()
    i = random() > 0.5
    
    p1 = load_player(config.model_1, config)
    print("You are playing against", config.model_1)
    print("Playing games with %d simulations per move" % config.simulation_num_per_move)
    p2 = HumanPlayer()
    side = -1
    while not game.game_over():
        if i:
            if side == -1:
                t = p1.pick_move(game, side)
            else:
                t = p2.pick_move(game, side)
        else:
            if side == 1:
                t = p1.pick_move(game, side)
            else:
                t = p2.pick_move(game, side)
        game.play_move(t[0], t[1], side)
        side *= -1
    print("\n\nFinal Board:")
    game.print_board()
    if (i and -1 == game.get_winner()) or (not i and 1 == game.get_winner()):
        print("You lose!")
    elif game.get_winner() == 0:
        print("Its a draw!")
    else:
        print("You win!")

def load_player(player_name, config):
    if player_name == "random":
        model = "random"
        player = RandomPlayer()
    elif player_name == "newest":
        model = sorted(glob.glob(config.data.model_location+"*.h5"))[-1]
        player = AIPlayer(0, config.simulation_num_per_move, train=False, weights=model)
    else:
        model = config.data.model_location+player_name
        player = AIPlayer(0, config.simulation_num_per_move, train=False, weights=model)
    return player