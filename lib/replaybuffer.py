from collections import deque
import random
try:
   import _pickle as pickle
except:
   import pickle

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, experience):
        self.buffer.append(experience)
        while len(self.buffer) > self.buffer_size: 
            self.buffer.popleft()
        
    def merge(self, buffer):
        for elem in buffer:
            self.add(elem)
        
    def size(self):
        return len(self.buffer)

    def sample(self, sample_size):
        sample = []

        if len(self.buffer) < sample_size:
            sample = random.sample(self.buffer, len(self.buffer))
        else:
            sample = random.sample(self.buffer, sample_size)

        return sample

    def save(self, filename):
        file = open(filename, 'wb') 
        pickle.dump(pickle.dumps(self.buffer), file)
        file.close() 
    
    def load(self, filename):
        try:
            file = open(filename, 'rb') 
            if len(self.buffer) == 0:
                self.buffer = pickle.loads(pickle.load(file))
            else:
                buf = pickle.loads(pickle.load(file))
                self.merge(buf)
            file.close() 
            return True
        except Exception as e:
            return False

    def clear(self):
        self.buffer.clear()