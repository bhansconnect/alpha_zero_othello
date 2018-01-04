from config import SelfPlayConfig
from lib import tf_util, util
from player.aiplayer import AIPlayer
from othello import Othello
from time import time
import glob
import os

def start():
    config = SelfPlayConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    util.set_low_process_priority()
    AIPlayer.create_if_nonexistant(config)
    run_games(config)
    
def run_games(config):
    game = Othello()
    model = ""
    x = config.iterations
    while(x != 0):
        x -= 1
        i = len(glob.glob(config.data.game_location+"*.pickle")) + 1
        print("Iteration %04d"%i)
        models = sorted(glob.glob(config.data.model_location+"*.h5"))
        if model == "":
            model = models[-1]
            print("Loading new model: %s" % model)
            ai = AIPlayer(config.buffer_size, config.game.simulation_num_per_move, model=model)
        elif models[-1] != model:
            model = models[-1]
            print("Loading new model: %s" % model)
            ai.load(model)
		
        start=time()
        for j in range(config.nb_game_in_file):
            util.progress(j, config.nb_game_in_file, start=start)
            side = -1
            turn = 1
            while not game.game_over():
                ai.tau = config.game.tau_1
                if config.game.tau_swap < turn:
                    ai.tau = config.game.tau_2
                t = ai.pick_move(game, side)
                game.play_move(t[0], t[1], side)
                side *= -1
                turn += 1
            ai.update_buffer(game.get_winner())
            game.reset_board()
        #print("Average Game Time: ", (time()-start)/(config.nb_game_in_file))
        util.progress(config.nb_game_in_file, config.nb_game_in_file, start=start)
        save_games(config, ai.buffer)
    t.join()
    
                
def save_games(config, buffer):
    buffer.save(config.data.game_location+str(time())+".pickle")
    games = glob.glob(config.data.game_location+"*.pickle")
    if len(games) > config.max_file_num:
        for file in sorted(games)[:-1*config.max_file_num]:
            os.remove(file)