"""
Created on Thu Sep 15 2022
@author: John Freeman

Wolf class, follows pheremone trails and eats sheep
reproduces rarely
dies slowly

population is difficult to maintain, tend to get stuck in pheromone spikes and die off
"""

import numpy as np

class Wolf():
    """Wolf class, takes in a shared env and pheremonal grid and optional x/y init"""
    def __init__(self, env, phers, x_coord=None, y_coord=None):
        if x_coord is None:
            self.x_coord = np.random.randint(0,len(env))
        else:
            self.x_coord = x_coord
        if y_coord is None:
            self.y_coord = np.random.randint(0,len(env))
        else:
            self.y_coord = y_coord
        self.env = env
        self.store = 1999
        self.wolf = True
        self.agent_refs = []
        self.active = True
        self.phers = phers

    def dist(self, agent2):
        """Distance between itself and a second agent"""
        return ((self.x_coord - agent2.x_coord)**2 + (self.y_coord - agent2.y_coord) **2)**0.5

    def reproduce(self, neighborhood):
        """Reproduce when another wolf is in proximity and both wolves are very full"""
        for agent in self.agent_refs:
            if not agent.wolf:
                continue
            if 0 < self.dist(agent) < neighborhood and \
                np.random.rand() < 0.001 and \
                agent.active and agent.store > 2350 and \
                self.store > 2350:

                new_x_coord = (self.x_coord + agent.x_coord) // 2
                new_y_coord = (self.y_coord + agent.y_coord) // 2

                if (new_y_coord == self.y_coord and new_x_coord == self.x_coord) or \
                   (new_y_coord == agent.y_coord and agent.x_coord == new_x_coord):
                    return
                self.agent_refs.append(Wolf(self.env, self.phers, x_coord=new_x_coord, y_coord=new_y_coord))
                break

    def eat(self):
        """Eat a sheep if it's on the same location as the sheep, mark the sheep as inactive. If starving, die."""
        if self.store > 2500:
            return
        if self.phers[self.y_coord][self.x_coord] >= 0.5: # check to make sure we don't iterate every single time
            for agent in self.agent_refs:
                if not agent.wolf and agent.active and agent.x_coord == self.x_coord and agent.y_coord == self.y_coord:
                    self.store += 2000
                    agent.active = False
                    break

        if self.store <= 0:
            # die
            self.active = False

    def move_local_grad(self):
        """Follow the local pheremone grid, consume some of the food stores"""
        left = (self.x_coord - 1) % len(self.phers)
        right = (self.x_coord + 1) % len(self.phers)
        upper = (self.y_coord - 1) % len(self.phers)
        lower = (self.y_coord + 1) % len(self.phers)
        left_grad = self.phers[self.y_coord][left] - self.phers[self.y_coord][self.x_coord]
        right_grad = self.phers[self.y_coord][right] - self.phers[self.y_coord][self.x_coord]
        up_grad = self.phers[upper][self.x_coord] - self.phers[self.y_coord][self.x_coord]
        down_grad = self.phers[lower][self.x_coord] - self.phers[self.y_coord][self.x_coord]

        if up_grad == down_grad == left_grad == right_grad:
            self.move_random()
            return

        max_grad = max([left_grad, right_grad, up_grad, down_grad])
        if left_grad == max_grad:
            self.x_coord = left
        elif right_grad == max_grad:
            self.x_coord = right
        elif down_grad == max_grad:
            self.y_coord = lower
        elif up_grad == max_grad:
            self.y_coord = upper

        self.store -= 25
        self.phers[self.y_coord][self.x_coord] = 0 # let them explore more

    def move_random(self):
        """Randomly move, used if no pheremone trail detected"""
        if np.random.rand() < 0.5:
            self.x_coord -= 1
        else:
            self.x_coord += 1

        if np.random.rand() < 0.5:
            self.y_coord -= 1
        else:
            self.y_coord += 1

        self.x_coord %= len(self.env)
        self.y_coord %= len(self.env)
        self.store -= 25
