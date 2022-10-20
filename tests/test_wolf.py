"""
Created on Thu Sep 15 2022
@author: John Freeman

Unit tests for the wolf agent. 
"""


import pytest
import numpy as np
from unittest import mock

from ..wolf import Wolf
from ..sheep import Sheep

BASE_ENV = np.array([[1,2,3],[4,5,6]])
BASE_PHERS = np.zeros_like(BASE_ENV)

class Testwolf:
    def test_dist(self):
        wolf1 = Wolf(BASE_ENV, BASE_PHERS, x_coord=5, y_coord=5)
        wolf2 = Wolf(BASE_ENV, BASE_PHERS, x_coord=10, y_coord=10)
        assert wolf1.dist(wolf2) == 50**0.5
    def test_reproduce(self):
        wolf1 = Wolf(BASE_ENV, BASE_PHERS, x_coord=6, y_coord=6)
        wolf2 = Wolf(BASE_ENV, BASE_PHERS, x_coord=12, y_coord=12)
        wolf1.store = 2450
        wolf2.store = 2450
        agents = [wolf1, wolf2]
        wolf1.agent_refs = wolf2.agent_refs = agents

        with mock.patch('numpy.random.rand', lambda: 0):
            wolf1.reproduce(10)

        assert(len(agents) == 3)

    def test_eat(self):
        custom_env = np.array([[100,0],[50,0]])
        custom_phers = np.zeros_like(custom_env)
        custom_phers[0][0] = 1

        wolf1 = Wolf(custom_env, custom_phers, x_coord=0, y_coord=0)
        wolf1.store = 0
        sheep1 = Sheep(custom_env, custom_phers, x_coord=0, y_coord=0)
        agents = [wolf1, sheep1]
        wolf1.agent_refs = agents

        wolf1.eat()

        assert(wolf1.store == 2000)
        assert(sheep1.active == False)

    def test_move(self):
        custom_env = np.array([[0,100,0],[50,0,0],[0,0,0]])
        custom_phers = np.array([[50,100,60],[50,40,20],[60,10,80]])

        wolf1 = Wolf(custom_env, custom_phers, x_coord=1, y_coord=1)
        wolf1.store = 100
        agents = [wolf1]
        wolf1.agent_refs = agents

        # expect an upward move
        wolf1.move_local_grad()

        assert(wolf1.x_coord == 1)
        assert(wolf1.y_coord == 0)

        assert(custom_phers[0][1] == 0)
        assert(wolf1.store == 75)
