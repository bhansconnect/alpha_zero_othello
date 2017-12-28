from alpha_zero_othello.config import DataConfig
import matplotlib.pyplot as plt
import pickle
import glob

def start():
    config = DataConfig()
    histories = glob.glob(config.history_location+"*.pickle")
    data = {}
    for hist in histories:
        file = open(hist, 'rb') 
        h = pickle.loads(pickle.load(file))
        for k, v in h.items():
            if k not in data.keys():
                data[k] = []
            for item in v:
                data[k].append(item)
    for i, kv in enumerate(data.items()):
        plt.subplot(1, len(data), i+1)
        plt.title(kv[0])
        plt.plot(kv[1])
    plt.show()
        