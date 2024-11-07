from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from model4_nosensors import WarehouseModel2, NormalWall, StorageWall, StorageZone, WaitPoint
from agent_nosensors_model4 import Forklift2

def agent_portrayal(agent):
    portrayal = {}
    if isinstance(agent, Forklift2):
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
                      "Color": "Blue"}],
                    data_collector_name='datacollector')

server = ModularServer(WarehouseModel2,
                       [grid, chart],
                       "Warehouse Model",
                       {"width": 30, "height": 30, "max_cycles": 10})

server.port = 8521
server.launch()
