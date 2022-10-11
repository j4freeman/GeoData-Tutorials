import pytest
import numpy as np
from unittest import mock

from ..sheep import Sheep

BASE_ENV = np.array([[1,2,3],[4,5,6]])
BASE_PHERS = np.zeros_like(BASE_ENV)

class TestSheep:
    def test_dist(self):
        sheep1 = Sheep(BASE_ENV, BASE_PHERS, x_coord=5, y_coord=5)
        sheep2 = Sheep(BASE_ENV, BASE_PHERS, x_coord=10, y_coord=10)
        assert sheep1.dist(sheep2) == 50**0.5
    def test_reproduce(self):
        sheep1 = Sheep(BASE_ENV, BASE_PHERS, x_coord=6, y_coord=6)
        sheep2 = Sheep(BASE_ENV, BASE_PHERS, x_coord=12, y_coord=12)
        sheep1.store = 150
        sheep2.store = 150
        agents = [sheep1, sheep2]
        sheep1.agent_refs = sheep2.agent_refs = agents

        with mock.patch('numpy.random.rand', lambda: 0):
            sheep1.reproduce(10)

        assert(len(agents) == 3)

        sheep3 = Sheep(BASE_ENV, BASE_PHERS, x_coord=80, y_coord=80)
        sheep3.store = 150
        sheep1.agent_refs.append(sheep3)
        sheep3.agent_refs = sheep1.agent_refs
        with mock.patch('numpy.random.rand', lambda: 0):
            sheep3.reproduce(10)

        assert(len(agents) == 4)

    def test_eat(self):
        custom_env = np.array([[100,0],[50,0]])
        custom_phers = np.zeros_like(custom_env)

        sheep1 = Sheep(custom_env, custom_phers, x_coord=0, y_coord=0)
        sheep1.store = 0
        agents = [sheep1]
        sheep1.agent_refs = agents

        sheep1.eat()

        assert(sheep1.store == 100)
        assert(custom_env[0][0] == 0)

        sheep1.y_coord = 1

        sheep1.eat()

        assert(sheep1.store == 150)
        assert(custom_env[0][0] == 0)

        sheep1.store = 0

        sheep1.eat()

        assert(sheep1.active == False)

    def test_move(self):
        custom_env = np.array([[0,100,0],[50,0,0],[0,0,0]])
        custom_phers = np.zeros_like(custom_env)

        sheep1 = Sheep(custom_env, custom_phers, x_coord=1, y_coord=1)
        sheep1.store = 50
        agents = [sheep1]
        sheep1.agent_refs = agents

        # expect an upward move
        sheep1.move_local_grad()

        assert(sheep1.x_coord == 1)
        assert(sheep1.y_coord == 0)

        assert(custom_phers[0][1] == 1) # remember inverted
        assert(sheep1.store == 0)
