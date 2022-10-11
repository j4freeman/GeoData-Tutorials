"""
Sheep class, simple agent that follows food gradients and eats while emitting pheremones
reproduces frequently
dies easily
"""

import numpy as np

class Sheep():
    """Accepts reference to an environmental raster, shared pheremone grid, and an optional x/y init location"""
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
        self.store = 450
        self.agent_refs = []
        self.active = True
        self.phers = phers
        self.wolf = False

    def dist(self, agent2):
        """Compare distance between itself and another agent"""
        return ((self.x_coord - agent2.x_coord)**2 + (self.y_coord - agent2.y_coord) **2)**0.5

    def reproduce(self, neighborhood):
        """Reproduce based on proximity, eating status, and random chance. 
           Offspring initialized halfway between two parents."""
        for agent in self.agent_refs:
            if agent.wolf:
                continue
            if 0 < self.dist(agent) < neighborhood and np.random.rand() < 0.001 and agent.active and self.store > 100:
                new_x_coord = (self.x_coord + agent.x_coord) // 2
                new_y_coord = (self.y_coord + agent.y_coord) // 2

                if (new_y_coord == self.y_coord and new_x_coord == self.x_coord) or \
                   (new_y_coord == agent.y_coord and agent.x_coord == new_x_coord):
                    return
                self.agent_refs.append(Sheep(self.env, self.phers, x_coord=new_x_coord, y_coord=new_y_coord))
                break

    def eat(self):
        """If sheep is full, do nothing. If there is food to eat, eat it. If starving, die."""
        if self.store > 450:
            return
        if self.env[self.y_coord][self.x_coord] > 100:
            self.env[self.y_coord][self.x_coord] -= 100
            self.store += 100

        else:
            self.store += self.env[self.y_coord][self.x_coord]
            self.env[self.y_coord][self.x_coord] = 0

        if self.store <= 0:
            # die
            self.active = False

    def move_local_grad(self):
        """Determine the direction of maximum food, go there, and reduce the store of food in the process."""
        left = (self.x_coord - 1) % len(self.env)
        right = (self.x_coord + 1) % len(self.env)
        upper = (self.y_coord - 1) % len(self.env)
        lower = (self.y_coord + 1) % len(self.env)
        left_grad = self.env[self.y_coord][left] - self.env[self.y_coord][self.x_coord]
        right_grad = self.env[self.y_coord][right] - self.env[self.y_coord][self.x_coord]
        up_grad = self.env[upper][self.x_coord] - self.env[self.y_coord][self.x_coord]
        down_grad = self.env[lower][self.x_coord] - self.env[self.y_coord][self.x_coord]

        max_grad = max([left_grad, right_grad, up_grad, down_grad])
        if left_grad == max_grad:
            self.x_coord = left
        elif right_grad == max_grad:
            self.x_coord = right
        elif down_grad == max_grad:
            self.y_coord = lower
        elif up_grad == max_grad:
            self.y_coord = upper

        self.phers[self.y_coord][self.x_coord] += 1
        self.store -= 50
