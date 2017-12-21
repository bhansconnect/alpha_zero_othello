import os

def project_dir():
    return os.path.dirname(os.path.abspath(__file__))

def data_dir():
    return os.path.join(project_dir(), "data")

class DataConfig:
    game_location = data_dir()+'/'+"games"+"/"
    model_location = data_dir()+'/'+"models"+"/"
    history_location = data_dir()+'/'+"history"+"/"
    def __init__(self):
        if not os.path.exists(DataConfig.game_location):
            os.makedirs(DataConfig.game_location)
        if not os.path.exists(DataConfig.model_location):
            os.makedirs(DataConfig.model_location)
        if not os.path.exists(DataConfig.history_location):
            os.makedirs(DataConfig.history_location)

class GameConfig:
    simulation_num_per_move = 100
    tau_1 = 1
    tau_2 = 1e-2
    tau_swap = 10 # change to tau_2 after this many moves

class SelfPlayConfig:
    nb_game_in_file = 10
    buffer_size = 64 * nb_game_in_file
    max_file_num = 10000  # 50000
    iterations = -1 #-1 for infinite
    gpu_mem_fraction = 0.1
    data = DataConfig()
    game = GameConfig()


class OptimizerConfig:
    batch_size = 512
    buffer_size = 1000000
    save_model_cycles = 1
    min_game_files = 100
    min_new_game_files = 10
    iterations = -1 #-1 for infinite
    gpu_mem_fraction = 0.6
    verbose = 0 #0,1,2 like keras model.fit
    learning_rate1 = 1e-2
    iter2 = 200
    learning_rate2 = 1e-3
    iter3 = 300
    learning_rate3 = 1e-4
    data = DataConfig()

class EvaluateConfig:
    repeat_with_new_model = True
    gpu_mem_fraction = 0.1
    model_1 = "newest" # options: "newest", "random" or file name in model location
    model_2 = "1513790737.0385518.h5" # options: "newest", "random" or file name in model location
    data = DataConfig()
    game = GameConfig()
    game.simulation_num_per_move = 50
    game_num = 100
    
class RankingConfig:
    game_num_per_model = 100
    gpu_mem_fraction = 0.1
    model_skip = 2 # aka grab every xth model...1 being all models
    data = DataConfig()
    game = GameConfig()
    game.simulation_num_per_move = 50