from collections import deque
import random
try:
   import cPickle as pickle
except:
   import pickle

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, experience):
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        
    def merge(self, buffer):
        for elem in buffer:
            self.add(elem)
        
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return batch

    def save(self, filename):
        file = open(filename, 'wb') 
        pickle.dump(pickle.dumps(self.buffer), file)
        file.close() 
    
    def load(self, filename):
        file = open(filename, 'rb') 
        if len(self.buffer) == 0:
            self.buffer = pickle.loads(pickle.load(file))
        else:
            buf = pickle.loads(pickle.load(file))
            self.merge(buf)
        file.close() 

    def clear(self):
        self.buffer.clear()
        self.count = 0