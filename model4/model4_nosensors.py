from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent_nosensors_model4 import Forklift2, NormalWall, StorageWall, StorageZone, WaitPoint
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from scipy.stats import truncnorm


class WarehouseModel2(Model):
    def __init__(self, width, height, total_time_hours=None, max_cycles=None):
        super().__init__()
        self.grid = MultiGrid(width, height, False)  # Disable toroidal (wrap-around) grid
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={
                "Energy Consumption": lambda a: a.energy_consumption if isinstance(a, Forklift2) else None,
                "Total Distance": lambda a: a.total_distance if isinstance(a, Forklift2) else None,
                "Unloading Cycles": lambda a: a.unloading_cycles if isinstance(a, Forklift2) else None,
                "Total Time": lambda a: a.total_time if isinstance(a, Forklift2) else None
            }
        )

        # Generate Monte Carlo samples
        self.generate_monte_carlo_samples()

        # Print Monte Carlo samples
        self.print_samples()

        # Create one forklift and place it at a specific position (e.g., (0, 0))
        self.forklift = Forklift2(
            1, self,
            self.empty_speeds[0],
            self.loaded_speeds[0],
            self.load_times[0],
            self.unload_times[0],
            self.searching_speed[0]
        )
        self.schedule.add(self.forklift)
        start_x, start_y = 13, 7  # Set the specific starting position here
        self.grid.place_agent(self.forklift, (start_x, start_y))

        self.loading_slot_set = False  # Flag to track if loading slot is set
        self.max_time_seconds = total_time_hours * 3600 if total_time_hours is not None else None
        self.max_cycles = max_cycles

        # Draw warehouse layout
        self.draw_warehouse()

    def generate_monte_carlo_samples(self):
        # Define means and standard deviations
        v_empty_mean = 4.17
        sigma_empty = 0.4
        v_loaded_mean = 1.94
        sigma_loaded = 0.3
        t_load_mean = 150
        sigma_load = 50
        t_unload_mean = 150
        sigma_unload = 50
        v_search_mean = 0.28
        sigma_search = 0.1

        # Generate 1000 random samples for each factor
        self.empty_speeds = np.random.normal(v_empty_mean, sigma_empty, 1000000)
        self.loaded_speeds = np.random.normal(v_loaded_mean, sigma_loaded, 1000000)

        # Generate truncated normal distribution for load and unload times
        lower, upper = 30, np.inf  # Truncate at 30 and above
        self.load_times = truncnorm(
            (lower - t_load_mean) / sigma_load,
            (upper - t_load_mean) / sigma_load,
            loc=t_load_mean,
            scale=sigma_load
        ).rvs(1000000)

        self.unload_times = truncnorm(
            (lower - t_unload_mean) / sigma_unload,
            (upper - t_unload_mean) / sigma_unload,
            loc=t_unload_mean,
            scale=sigma_unload
        ).rvs(1000000)

        # Generate truncated normal distribution for searching speed
        lower_search, upper_search = 0.28, np.inf  # Truncate at 0.28 and above
        self.searching_speed = truncnorm(
            (lower_search - v_search_mean) / sigma_search,
            (upper_search - v_search_mean) / sigma_search,
            loc=v_search_mean,
            scale=sigma_search
        ).rvs(1000000)

    def print_samples(self):
        # Print the first 5 samples of each factor
        print("Empty Speeds (first 5 samples):", self.empty_speeds[:5])
        print("Loaded Speeds (first 5 samples):", self.loaded_speeds[:5])
        print("Load Times (first 5 samples):", self.load_times[:5])
        print("Unload Times (first 5 samples):", self.unload_times[:5])
        print("Search Speeds (first 5 samples):", self.searching_speed[:5])

        # Visualize the samples using histograms
        plt.figure(figsize=(10, 8))

        plt.subplot(3, 2, 1)
        plt.hist(self.empty_speeds, bins=30, edgecolor='black')
        plt.title('Empty Speeds')

        plt.subplot(3, 2, 2)
        plt.hist(self.loaded_speeds, bins=30, edgecolor='black')
        plt.title('Loaded Speeds')

        plt.subplot(3, 2, 3)
        plt.hist(self.load_times, bins=30, edgecolor='black')
        plt.title('Load Times')

        plt.subplot(3, 2, 4)
        plt.hist(self.unload_times, bins=30, edgecolor='black')
        plt.title('Unload Times')

        plt.subplot(3, 2, 5)
        plt.hist(self.searching_speed, bins=30, edgecolor='black')
        plt.title('Search Speeds')

        plt.tight_layout()
        plt.show()

    def draw_warehouse(self):
        ######## top left zone #########
        #first storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 28))
        #second storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 26))
        #third storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 21), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 24))
        #fourth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 19), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 22))
        #fifth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 17), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 20))
        #sixth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 18))
        #first waitpoint horizontal
        for x in range(0, 8):
            wall = WaitPoint((x, 21), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 27))
        #second waitpoint horizontal
        for x in range(0, 8):
            wall = WaitPoint((x, 21), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 25))
        #third waitpoint horizontal
        for x in range(0, 8):
            wall = WaitPoint((x, 21), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 23))
         #fourth waitpoint horizontal
        for x in range(0, 8):
            wall = WaitPoint((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 21))
         #fifth waitpoint horizontal
        for x in range(0, 8):
            wall = WaitPoint((x, 16), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 19))
        #First storage wall vertical
        for y in range(18, 30):
            wall = StorageWall((15, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (15, y))
        # Second storage wall vertical
        for y in range(18, 30):
            wall = StorageWall((17, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (17, y))
        # First waitpoint vertical
        for y in range(18, 30):
            wall = WaitPoint((16, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (16, y))
        # Vertical wall
        for y in range(14, 18):
            wall = NormalWall((17, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (18, y))
        # Vertical wall
        for y in range(0, 3):
            wall = NormalWall((17, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (18, y))
        # Second vertical wall
        for y in range(5, 12):
            wall = NormalWall((17, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (18, y))
        # third vertical wall
        for y in range(18, 29):
            wall = NormalWall((14, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (14, y))
        ########## low left zone ################
        #low left first vertical wall
        for y in range(13, 17):
            wall = NormalWall((8, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (7, y))
        #low left second vertical wall
        for y in range(0, 10):
            wall = NormalWall((8, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (7, y))
        #first storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 16))
        #second storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 14))
        #third storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 12))
        #fourth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 10))
        #fifth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 8))
        #sixth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 6))
        #seventh storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 4))
        #eight storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 2))
        #nineth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 0))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 15))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 13))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 11))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 9))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 7))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 5))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 3))
        #first waitpointlow left
        for x in range(0, 5):
            wall = WaitPoint((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 1))
        #first vertical wall low left side
        for y in range(18, 29):
            wall = NormalWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (18, y))
        # first horizontal wall
        for x in range(0, 8):
            wall = NormalWall((x, 10), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 17))
        # boundary horizontal wall top right
        for x in range(18, 30):
            wall = NormalWall((x, 26), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 29))
        # boundary horizontal wall top left
        for x in range(0, 14):
            wall = NormalWall((x, 26), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 29))
        ######### low right zone ############
        # boundary horizontal wall low right - top
        for x in range(18, 30):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 11))
        # boundary horizontal wall low right - low
        for x in range(18, 30):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 0))
        #boundary vertical wall low right
        for y in range(0, 11):
            wall = NormalWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (29, y))
        #first storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (20, y))
        #second storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (22, y))
        #third storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (24, y))
        #fourth storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (26, y))
        #fifth storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (28, y))
        # storage wall low right - long
        for x in range(19, 29):
            wall = StorageWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 1))
        #first waitpoint low right
        for y in range(5, 11):
            wall = WaitPoint((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (19, y))
        #second waitpoint low right
        for y in range(5, 11):
            wall = WaitPoint((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (21, y))
        #third waitpoint low right
        for y in range(5, 11):
            wall = WaitPoint((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (23, y))
        #fourth waitpoint low right
        for y in range(5, 11):
            wall = WaitPoint((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (25, y))
        #fifth waitpoint low right
        for y in range(5, 11):
            wall = WaitPoint((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (27, y))
        #sixth waitpoint right - long
        for x in range(19, 29):
            wall = WaitPoint((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 2))
        # Add unloading zone
        for x in range(19, 23):
            for y in range(16, 20):
                storage_zone = StorageZone((x, y), self)
                self.schedule.add(storage_zone)
                self.grid.place_agent(storage_zone, (x, y))

    def randomize_loading_slot(self):
        # Reset any existing red waitpoints to yellow
        for agent in self.schedule.agents:
            if isinstance(agent, WaitPoint) and agent.color == "red":
                agent.color = "yellow"

        # Find all waitpoint agents
        waitpoint_agents = [agent for agent in self.schedule.agents if isinstance(agent, WaitPoint)]
        if waitpoint_agents:
            # Choose a random waitpoint agent
            random_waitpoint = self.random.choice(waitpoint_agents)
            random_waitpoint.color = "red"
            self.loading_slot_set = True  # Mark that the loading slot has been set

    def reset_loading_slot(self):
        self.loading_slot_set = False
        self.randomize_loading_slot()

    def get_total_time(self):
        return self.forklift.total_time

    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)
        total_time = self.get_total_time()

        # Only set the loading slot once
        if not self.loading_slot_set:
            self.randomize_loading_slot()

        # Print the total time and energy consumption
        formatted_time = self.format_time(total_time)
        energy_consumption = self.forklift.calculate_energy_consumption()
        print(f"Total Time: {formatted_time}")  # Print total time in a readable format
        print(f"Energy Consumption: {energy_consumption:.2f} kWh")  # Print energy consumption

        # Check if the forklift has completed the specified total time or cycles
        if (self.max_time_seconds is not None and total_time >= self.max_time_seconds) or (self.max_cycles is not None and self.forklift.unloading_cycles >= self.max_cycles):
            self.running = False  # Stop the simulation
