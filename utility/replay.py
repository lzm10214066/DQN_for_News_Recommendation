'''
modified by zhenmaoli
2018.3.16
reference: reference:http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
'''

from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size,random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self,T):
        if self.count < self.buffer_size:
            self.buffer.append(T)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(T)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch,  s2_batch,t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0