from collections import namedtuple, deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition', 
                                   ('currentState', 'action', 'nextState', 'reward'))
        
    def push(self, *args):
        self.memory.append(self.Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory) 