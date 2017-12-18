from alpha_zero_othello.config import EloConfig
from alpha_zero_othello.lib import tf_util, util
from alpha_zero_othello.player.aiplayer import AIPlayer
from alpha_zero_othello.othello import Othello
from time import time
import numpy as np
import random
import glob

def start():
    config = EloConfig()
    tf_util.update_memory(config.gpu_mem_fraction)

    
    models = glob.glob(config.data.model_location+"*.h5")
    if len(models) == 0:
        ai = AIPlayer(config.buffer_size, config.simulation_num_per_move)
        ai.save_weights(config.data.model_location+str(time())+".h5")
    calc_elo(config)
    
def calc_elo(config):
    models = glob.glob(config.data.model_location+"*.h5")
    players = []
    for i, model in enumerate(models):
        if i % config.model_skip == 0 or i == len(models):
            players.append(model)
    
    elo = np.zeros((len(players)))
    game = Othello()
    
    ##give every player a random order to play games against opponents
    order = []
    for i in range(len(players)):
        nums = [x for x in range(len(players))]
        nums.remove(i)
        order.append(random.shuffle(nums))
    
    for i in config.game_num_per_model:
        print("Playing game #%d for each player" % (i+1))
        start = time()
        for j in range(len(players)):
            util.progress(j, len(players), start=start)
            if i == len(players) - 1:
                order[j] = random.shuffle(order[j])
            if i >= len(players) -1:
                i %= config.game_num_per_model
            p1 = AIPlayer(1, config.simulation_num_per_move, weights=players[j])
            p2 = AIPlayer(1, config.simulation_num_per_move, weights=players[order[j][i]])
            side = -1
            while not game.game_over():
                if side == -1:
                    t = p1.pick_move(game, side)
                else:
                    t = p2.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
            if game.get_winner() == -1:
                elo[j] += config.k_val*(1-expected(elo[j], elo[order[j][i]]))
                elo[order[j][i]] += config.k_val*(0-expected(elo[order[j][i]], elo[j]))
            elif game.get_winner() == 1:
                elo[j] += config.k_val*(0-expected(elo[j], elo[order[j][i]]))
                elo[order[j][i]] += config.k_val*(1-expected(elo[order[j][i]], elo[j]))
            else:
                elo[j] += config.k_val*(0.5-expected(elo[j], elo[order[j][i]]))
                elo[order[j][i]] += config.k_val*(0.5-expected(elo[order[j][i]], elo[j]))
            if(elo[j] < 0):
                elo[j] = 0
            if(elo[order[j][i]] < 0):
                elo[order[j][i]] = 0
        util.progress(len(players), len(players), start=start)
        print(elo)
        print(np.argsort(elo))
            
def expected(ra, rb):
    dif = ra - rb
    denom = 1 + 10**(dif/400)
    return 1/denom