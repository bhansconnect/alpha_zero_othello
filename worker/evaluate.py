from config import EvaluateConfig
from lib import tf_util, util
from player.player import RandomPlayer
from player.aiplayer import AIPlayer
from othello import Othello
from time import time, sleep
import glob

def start():
    config = EvaluateConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    AIPlayer.create_if_nonexistant(config)
    run_games(config)
    
def run_games(config):
    game = Othello()
    model_1 = ""
    model_2 = ""
    p1, new_1 = create_player(config.model_1, model_1, config)
    p2, new_2 = create_player(config.model_2, model_2, config)
    if config.model_1 == "newest" or config.model_2 == "newest":
        i = len(glob.glob(config.data.model_location+"*.h5"))
    else:
        i = 0
    avg_wins = []
    while True:
        i += 1
        new_1 = load_player(p1, config.model_1, model_1, config)
        new_2 = load_player(p2, config.model_2, model_2, config)
        while((config.model_1 == "newest" and new_1 == model_1) or (config.model_2 == "newest" and new_2 == model_2)):
            #print("Waiting on new model. Sleeping for 1 minute.")
            sleep(60)
            new_1 = load_player(p1, config.model_1, model_1, config)
            new_2 = load_player(p2, config.model_2, model_2, config)
        model_1 = new_1
        model_2 = new_2
        wins = 0
        losses = 0
        ties = 0
        print("Iteration %04d"%i)
        print("Playing games between %s and %s" % (config.model_1, config.model_2))
        print("Playing %d games with %d simulations per move" % (config.game_num, config.game.simulation_num_per_move))
        start=time()
        for j in range(config.game_num):
            util.print_progress_bar(j, config.game_num, start=start)
            side = -1
            turn = 1
            while not game.game_over():
                tau = config.game.tau_1
                if config.game.tau_swap < turn:
                    tau = config.game.tau_2
                if config.model_1 != "random":
                    p1.tau =tau
                if config.model_2 != "random":
                    p2.tau = tau
                if j % 2 == 0:
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
                turn += 1
            if game.get_winner() == 0:
                ties += 1
                savePerformance(config, model_1, model_2, 0, 1, 0)
            elif (j % 2 == 0 and game.get_winner() == -1) or (j % 2 == 1 and game.get_winner() == 1):
                wins += 1
                savePerformance(config, model_1, model_2, 1, 0, 0)
            else:
                losses += 1
                savePerformance(config, model_1, model_2, 0, 0, 1)
            game.reset_board()
        util.print_progress_bar(config.game_num, config.game_num, start=start)
        print("%s vs %s: (%0.2f%% wins|%0.2f%% ties|%0.2f%% losses) of %d games" % (config.model_1, config.model_2, 
              100*wins/config.game_num, 100*ties/config.game_num, 100*losses/config.game_num, config.game_num))
        avg_wins.append(100*wins/config.game_num)
        if len(avg_wins) > config.rolling_avg_amount:
            avg_wins = avg_wins[-1*config.rolling_avg_amount:]
        print("Average Win Percent: %0.2f%%" % (sum(avg_wins)/float(len(avg_wins))))
        
        if not (config.repeat_with_new_model and (config.model_1 == "newest" or config.model_2 == "newest")):
            break
        
def create_player(player_name, current, config):
    if player_name == "random":
        model = "random"
        player = RandomPlayer()
    elif player_name == "newest":
        model = sorted(glob.glob(config.data.model_location+"*.h5"))[-1]
        if model != current:
            print("Loading new model: %s" % model)
        player = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=model, tau=config.game.tau_1)
    else:
        model = config.data.model_location+player_name+".h5"
        player = AIPlayer(0, config.game.simulation_num_per_move, train=False, model=model, tau=config.game.tau_1)
    return player, model
    
def load_player(player, player_name, current, config):
    if player_name == "newest":
        model = sorted(glob.glob(config.data.model_location+"*.h5"))[-1]
        if model != current:
            print("Loading new model: %s" % model)
            player.load(model)
        return model
    else:
        return current
    
def savePerformance(config, model_1, model_2, wins, ties, losses):
    m1 = config.model_1
    if m1 == "newest":
        m1 = util.getPlayerName(model_1)
    m2 = config.model_2
    if m2 == "newest":
        m2 = util.getPlayerName(model_2)
    util.saveWTL(config, m1, m2, wins, ties, losses)
    util.mergeStagedWTL(config)