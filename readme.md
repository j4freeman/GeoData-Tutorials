## Leeds Assignment 1 - Wolves and Sheep

Simple ABM model operating on an inital food raster that regenerates over time. Two types of agents, sheep consume nutrients from the raster, emit pheremones, and move in the direction of their local food gradient. Wolves follow pheremone trails and eat sheep. Pheremones diffuse via a gaussian filter at each interval. 

To execute, first install the requirements via running:
```
pip install -r requirements.txt
```

Then run:
```
python main.py
```

Note you can also pass optional arguments, in the order num_agents, num_iters, neighborhood, ie:
```
python main.py 100 100 20
```

To run tests, run:
```
pytest tests
```