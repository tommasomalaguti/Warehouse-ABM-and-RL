from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model1_sensors import WarehouseModel, NormalWall, StorageWall, StorageZone, WaitPoint
from agent_sensors_model1 import Forklift

def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, Forklift):
        portrayal = {
            "Shape": "circle",
            "Color": "red" if agent.currently_loaded else "blue",
            "Filled": "true",
            "Layer": 2,  # Ensure forklift is on top layer
            "r": 0.5
        }
    elif isinstance(agent, NormalWall):
        portrayal = {
            "Shape": "rect",
            "Color": agent.color,
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, StorageWall):
        portrayal = {
            "Shape": "rect",
            "Color": agent.color,
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, StorageZone):
        portrayal = {
            "Shape": "rect",
            "Color": "green",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, WaitPoint):
        portrayal = {
            "Shape": "rect",
            "Color": agent.color,
            "Filled": "true",
            "Layer": 0,  # Ensure waitpoint is on a lower layer
            "w": 1,
            "h": 1
        }
    return portrayal

grid = CanvasGrid(agent_portrayal, 45, 30, 800, 500)
chart = ChartModule([{"Label": "Energy Consumption",
                      "Color": "Black"},
                     {"Label": "Total Distance",
                      "Color": "Blue"},
                     {"Label": "Total Time",
                      "Color": "Red"},
                     {"Label": "Unloading Cycles",
                      "Color": "Green"}],
                    data_collector_name='datacollector')

server = ModularServer(WarehouseModel,
                       [grid, chart],
                       "Warehouse Model",
                       {"width": 30, "height": 30, "max_cycles": 10})

server.port = 8521
server.launch()
