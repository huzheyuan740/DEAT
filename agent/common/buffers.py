from collections import namedtuple
import random

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ExperiencePool:
    def __init__(self, experience_pool_size):
        self.experience_pool_size = experience_pool_size
        self.experience_pool_list = []
        self.experience_pool_index = 0

    def __len__(self):
        return len(self.experience_pool_list)

    def push(self, *args):
        temp_experience = Experience(*args)
        if len(self.experience_pool_list) < self.experience_pool_size:
            self.experience_pool_list.append(temp_experience)
        else:
            self.experience_pool_list[self.experience_pool_index] = temp_experience
        self.experience_pool_index = (self.experience_pool_index + 1) % self.experience_pool_size

    def sample(self, batch_size):
        # batch_size = min(batch_size, len(self.experience_pool_list))
        return random.sample(self.experience_pool_list, batch_size)

    def whether_full(self):
        return len(self.experience_pool_list) >= self.experience_pool_size
