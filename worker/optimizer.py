from config import OptimizerConfig
from lib import tf_util, util
from player.aiplayer import AIPlayer
from time import time, sleep
try:
   import _pickle as pickle
except:
   import pickle
import glob

def start():
    config = OptimizerConfig()
    tf_util.update_memory(config.gpu_mem_fraction)
    util.set_high_process_priority()
    AIPlayer.create_if_nonexistant(config)
    models = sorted(glob.glob(config.data.model_location+"*.h5"))
    ai = AIPlayer(config.buffer_size, 1, model=models[-1], compile=True)
    train(ai, config)
    
def train(ai, config):
    loaded_files = []
    x = config.iterations
    i = len(glob.glob(config.data.model_location+"*.h5"))
    while(x != 0):
        if i > config.iter3:
            ai.update_lr(config.learning_rate3)
        elif i > config.iter2:
            ai.update_lr(config.learning_rate2)
        else:
            ai.update_lr(config.learning_rate1)
        loaded_files = load_games(ai, loaded_files, config)
        length = 0
        start = time()
        print("Iteration %04d"%i)
        progress_start = (i-2)*config.min_new_game_files+config.min_game_files if i > 1 else 0
        progress_end = (i-1)*config.min_new_game_files+config.min_game_files
        util.print_progress_bar(0, progress_end-progress_start, start=start)
        while(len(loaded_files) < config.min_game_files or ((len(loaded_files)-config.min_game_files)//config.min_new_game_files)+1 < i):
            if length != len(loaded_files):
                length = len(loaded_files)
                util.print_progress_bar(length-progress_start, progress_end-progress_start, start=start)
            sleep(5)
            loaded_files = load_games(ai, loaded_files, config)
        util.print_progress_bar(progress_end-progress_start, progress_end-progress_start, start=start)
        print("Training for %d batches on %d samples" % (config.batches_per_iter, len(ai.buffer.buffer)))
        start = time()
        history = ai.train_batches(config.batch_size, config.batches_per_iter, config.verbose)
        for val in history.history.keys():
            print("%s: %0.4f" % (val, history.history[val][-1]))
        if i % config.save_model_cycles == 0:
            ai.save(config.data.model_location+str(time())+".h5")
			
        file = open(config.data.history_location+str(time())+".pickle", 'wb') 
        pickle.dump(pickle.dumps(history.history), file)
        file.close() 
        print("Iteration Time: %0.2f" % (time()-start))
        x -= 1
        i += 1

def load_games(ai, loaded_files, config):
    games = sorted(glob.glob(config.data.game_location+"*.pickle"))
    new = [game for game in games if game not in loaded_files]
    for game in sorted(new):
        ai.buffer.load(game)
    return games