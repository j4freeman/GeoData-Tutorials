"""
Main entry point for the ABM simulator, accepts standard input and initializes and exeutes the model
"""

import sys
from model import ABMModel

# either want all three arguments or no arguments
if len(sys.argv) == 4:
    NUM_AGENTS = int(sys.argv[1])
    NUM_ITERS = int(sys.argv[2])
    NEIGHBORHOOD = int(sys.argv[3])
elif len(sys.argv) == 1:
    NUM_AGENTS = 100
    NUM_ITERS = 10000
    NEIGHBORHOOD = 20
else:
    sys.exit(-1)

RASTER_NAME = "in.txt"
COORDS_PAGE = "https://www.geog.leeds.ac.uk/courses/computing/practicals/python/agent-framework/part9/data.html"

if __name__ == "__main__":
    model = ABMModel(NUM_AGENTS, NUM_ITERS, NEIGHBORHOOD, RASTER_NAME, COORDS_PAGE)
    model.run()
