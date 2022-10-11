"""Model framework class housing code to build the environment grids and run the agents' loops"""

import tkinter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation
import scipy
import scipy.ndimage
import pandas as pd

from sheep import Sheep
from wolf import Wolf

matplotlib.use('TkAgg')

class ABMModel():
    """
    Takes in number of agents, number of iterations to simulate, neighborhood threshold for reproduction,
    a file name of the input raster, and a http link to the list of coordinates. Initializes the agents list
    and environment based on these values."""
    def __init__(self, num_agents, num_iters, neighborhood, raster_name, coords_page):
        self.num_agents = num_agents
        self.num_iters = num_iters
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.neighborhood = neighborhood
        self.init_env(raster_name)
        self.init_agents_list(coords_page)

    def init_env(self, raster_name):
        """Takes in the raster file path, reads it and constructs the env and pheremone matrices"""
        data = []
        with open(raster_name) as f:
            for line in f:
                parsed_line = str.split(line, ",")
                data_line = []
                for word in parsed_line:
                    data_line.append(float(word))
                data.append(data_line)

        self.data = np.array(data)
        self.phers = np.zeros_like(data)

    def init_agents_list(self, coords_page):
        """Read from the coordinates web page and create agents up to num_agents. If more agents needed than specified
        in the web page, generate further ones randomly. """
        self.agents = []
        if coords_page is None:
            agent_coords = pd.read_html(coords_page)[0]
        else:
            agent_coords = pd.DataFrame()
        if agent_coords.shape[0] > 0:
            agent_coords["x"] = agent_coords["x"].astype(int)
            agent_coords["y"] = agent_coords["y"].astype(int)

        for i in range(self.num_agents):
            if i < agent_coords.shape[0]:
                x = agent_coords.iloc[i]["x"]
                y = agent_coords.iloc[i]["y"]
            else:
                x = None
                y = None
            if np.random.rand() < 0.25:
                self.agents.append(Wolf(self.data, self.phers, x_coord=x, y_coord=y))
            else:
                self.agents.append(Sheep(self.data, self.phers, x_coord=x, y_coord=y))
            self.agents[-1].agent_refs = self.agents

    def gen(self):
        """Generator function for input to the animation"""
        i = 0
        while i <= self.num_iters:
            i += 1
            yield i

    def update(self, frame):
        """Main function handling animations.
        Move the agents, have them eat and reproduce, regrow vegetation on the main grid and diffuse
        pheremones via a gaussian filter. Remove agents that have died from the main agent list and return plot
        of relevant data as output. """
        self.fig.clear()
        print(f"""iter: {frame},
                  total sheep is: {len([x for x in self.agents if not x.wolf])},
                  total wolves: {len([x for x in self.agents if x.wolf])},
                  total store is: {sum(x.store for x in self.agents if x.active)},
                  total env is: {self.data.sum()}""")
        np.random.shuffle(self.agents)
        for agent in self.agents:
            if not agent.active:
                continue
            agent.eat()
            agent.move_local_grad()
            agent.reproduce(self.neighborhood)
            if agent.wolf: # let them move a bit faster
                agent.eat()
                agent.move_local_grad()
        self.data += 1

        # diffuse pheromone trails
        # this whole process breaks down pretty quickly for large numbers of agents and gets saturated
        # also doesn't diffuse around edges of the map
        gauss = scipy.ndimage.gaussian_filter(self.phers, [1,1], mode='constant')
        self.phers += gauss
        self.phers /= 2

        for idx, agent in enumerate(self.agents):
            if not agent.active:
                self.agents.pop(idx)

        ax = self.fig.add_subplot(121)
        ax.imshow(self.data)
        ax.scatter([x.x_coord for x in self.agents if not x.wolf],
                   [x.y_coord for x in self.agents if not x.wolf],
                   c='white', s=2, marker="1")
        ax.scatter([x.x_coord for x in self.agents if x.wolf],
                   [x.y_coord for x in self.agents if x.wolf],
                   c='red', s=3, marker="1")
        ax.title.set_text("Food Map")

        ax = self.fig.add_subplot(122)
        ax.imshow(self.phers)
        ax.scatter([x.x_coord for x in self.agents if not x.wolf],
                   [x.y_coord for x in self.agents if not x.wolf],
                   c='white', s=1, marker="1")
        ax.scatter([x.x_coord for x in self.agents if x.wolf],
                   [x.y_coord for x in self.agents if x.wolf],
                   c='red', s=2, marker="1")
        ax.title.set_text("Pheromone Map")

        self.fig.suptitle(f"""Total Sheep: {len([x for x in self.agents if not x.wolf])},
                              Total Wolves: {len([x for x in self.agents if x.wolf])},
                              Total Pheremones: {self.phers.sum()}""")

    def run_anim(self):
        """Run function wrapping the matplotlib animator"""
        anim = matplotlib.animation.FuncAnimation(self.fig, self.update, interval=1, frames=self.gen, repeat=False)
        self.canvas.draw()

    def run(self):
        """Main entry point, constructs the TK ui and waits for user input to execute"""
        root = tkinter.Tk()
        root.wm_title("Model")
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=root)
        self.canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        menu_bar = tkinter.Menu(root)
        root.config(menu=menu_bar)
        model_menu = tkinter.Menu(menu_bar)
        menu_bar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Run model", command=self.run_anim)
        tkinter.mainloop()
