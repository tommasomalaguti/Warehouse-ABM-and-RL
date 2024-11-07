from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent_nosensors_threeforks_model5 import Forklift2, NormalWall, StorageWall, StorageZone, WaitPoint
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from scipy.stats import truncnorm


class WarehouseModel2(Model):
    def __init__(self, width, height, total_time_hours=None, max_cycles=None):
        super().__init__()
        self.grid = MultiGrid(width, height, False)
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

        # Generate unique samples for each forklift
        forklift_samples = [
            (self.empty_speeds[:1000], self.loaded_speeds[:1000], self.load_times[:1000], self.unload_times[:1000], self.searching_speed[:1000]),
            (self.empty_speeds[1000:2000], self.loaded_speeds[1000:2000], self.load_times[1000:2000], self.unload_times[1000:2000], self.searching_speed[1000:2000]),
            (self.empty_speeds[2000:3000], self.loaded_speeds[2000:3000], self.load_times[2000:3000], self.unload_times[2000:3000], self.searching_speed[2000:3000])
        ]

        # Instantiate forklifts with their respective samples
        self.forklifts = [
            Forklift2(1, self, *forklift_samples[0], "red", "grey"),
            Forklift2(2, self, *forklift_samples[1], "orange", "green"),
            Forklift2(3, self, *forklift_samples[2], "purple", "blue")
        ]

        start_positions = [(13, 7), (14, 7), (12, 7)]
        for forklift, pos in zip(self.forklifts, start_positions):
            self.schedule.add(forklift)
            self.grid.place_agent(forklift, pos)

        self.red_loading_slot_set = False
        self.orange_loading_slot_set = False
        self.purple_loading_slot_set = False
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
        self.load_times = np.random.normal(t_load_mean, sigma_load, 1000000)
        self.unload_times = np.random.normal(t_unload_mean, sigma_unload, 1000000)

        # Generate truncated normal distribution for searching speed
        lower, upper = 0.28, np.inf  # Truncate at 0.28 and above
        self.searching_speed = truncnorm(
            (lower - v_search_mean) / sigma_search, 
            (upper - v_search_mean) / sigma_search, 
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
        zone = 1  # Assuming zone 1 for this section
        # First storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 28))
        # Second storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 26))
        # Third storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 21), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 24))
        # Fourth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 19), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 22))
        # Fifth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 17), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 20))
        # Sixth storage wall top left
        for x in range(0, 8):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 18))
        # First waitpoint horizontal
        for x in range(0, 8):
            waitpoint = WaitPoint((x, 21), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 27))
        # Second waitpoint horizontal
        for x in range(0, 8):
            waitpoint = WaitPoint((x, 21), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 25))
        # Third waitpoint horizontal
        for x in range(0, 8):
            waitpoint = WaitPoint((x, 21), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 23))
         # Fourth waitpoint horizontal
        for x in range(0, 8):
            waitpoint = WaitPoint((x, 18), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 21))
         # Fifth waitpoint horizontal
        for x in range(0, 8):
            waitpoint = WaitPoint((x, 16), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 19))
        # First storage wall vertical
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
            waitpoint = WaitPoint((16, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (16, y))
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
        # Third vertical wall
        for y in range(18, 29):
            wall = NormalWall((14, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (14, y))
        ########## top right zone ###############
        zone = 2  # Assuming zone 2 for this section
        # Boundary vertical wall top right
        for y in range(0, 30):
            wall = NormalWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (29, y))
        # Boundary horizontal wall top right - top
        for x in range(18, 23):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 20))
        # Boundary horizontal wall top right - top
        for x in range(25, 30):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 20))
        for x in range(19, 23):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 27))
        for x in range(25, 29):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 27))
        for x in range(19, 23):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 25))
        for x in range(25, 29):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 25))
        for x in range(19, 23):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 23))
        for x in range(25, 29):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 23))
        for x in range(19, 23):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 21))
        for x in range(25, 29):
            wall = StorageWall((x, 23), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 21))
        for x in range(19, 23):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 28))
        for x in range(25, 29):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 28))
        for x in range(19, 23):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 26))
        for x in range(25, 29):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 26))
        for x in range(19, 23):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 24))
        for x in range(25, 29):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 24))
        for x in range(19, 23):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 22))
        for x in range(25, 29):
            waitpoint = WaitPoint((x, 23), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 22))
        ########## low left zone ################
        zone = 3  # Assuming zone 3 for this section
        # Low left first vertical wall
        for y in range(13, 17):
            wall = NormalWall((8, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (7, y))
        # Low left second vertical wall
        for y in range(0, 10):
            wall = NormalWall((8, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (7, y))
        # First storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 16))
        # Second storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 14))
        # Third storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 12))
        # Fourth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 10))
        # Fifth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 8))
        # Sixth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 6))
        # Seventh storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 4))
        # Eighth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 2))
        # Ninth storage wall low left
        for x in range(0, 5):
            wall = StorageWall((x, 15), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 0))
        # First waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 15))
        # Second waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 13))
        # Third waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 11))
        # Fourth waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 9))
        # Fifth waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 7))
        # Sixth waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 5))
        # Seventh waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 3))
        # Eighth waitpoint low left
        for x in range(0, 5):
            waitpoint = WaitPoint((x, 15), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 1))
        # First vertical wall low left side
        for y in range(18, 29):
            wall = NormalWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (18, y))
        # First horizontal wall
        for x in range(0, 8):
            wall = NormalWall((x, 10), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 17))
        # Boundary horizontal wall top right
        for x in range(18, 30):
            wall = NormalWall((x, 26), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 29))
        # Boundary horizontal wall top left
        for x in range(0, 14):
            wall = NormalWall((x, 26), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 29))
        ######### low right zone ############
        zone = 4  # Assuming zone 4 for this section
        # Boundary horizontal wall low right - top
        for x in range(18, 30):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 11))
        # Boundary horizontal wall low right - low
        for x in range(18, 30):
            wall = NormalWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 0))
        # Boundary vertical wall low right
        for y in range(0, 11):
            wall = NormalWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (29, y))
        # First storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (20, y))
        # Second storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (22, y))
        # Third storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (24, y))
        # Fourth storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (26, y))
        # Fifth storage wall low right
        for y in range(5, 11):
            wall = StorageWall((18, y), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (28, y))
        # Storage wall low right - long
        for x in range(19, 29):
            wall = StorageWall((x, 18), self)
            self.schedule.add(wall)
            self.grid.place_agent(wall, (x, 1))
        # First waitpoint low right
        for y in range(5, 11):
            waitpoint = WaitPoint((18, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (19, y))
        # Second waitpoint low right
        for y in range(5, 11):
            waitpoint = WaitPoint((18, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (21, y))
        # Third waitpoint low right
        for y in range(5, 11):
            waitpoint = WaitPoint((18, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (23, y))
        # Fourth waitpoint low right
        for y in range(5, 11):
            waitpoint = WaitPoint((18, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (25, y))
        # Fifth waitpoint low right
        for y in range(5, 11):
            waitpoint = WaitPoint((18, y), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (27, y))
        # Sixth waitpoint right - long
        for x in range(19, 29):
            waitpoint = WaitPoint((x, 18), self, zone)
            self.schedule.add(waitpoint)
            self.grid.place_agent(waitpoint, (x, 2))
        # Add unloading zone
        for x in range(19, 23):
            for y in range(16, 20):
                storage_zone = StorageZone((x, y), self)
                self.schedule.add(storage_zone)
                self.grid.place_agent(storage_zone, (x, y))

    def randomize_loading_slot(self, color):
        waitpoints = [agent for agent in self.schedule.agents if isinstance(agent, WaitPoint) and agent.color == "yellow"]
        if color == "red" and not self.red_loading_slot_set:
            if waitpoints:
                random_waitpoint = self.random.choice(waitpoints)
                random_waitpoint.color = "red"
                self.red_loading_slot_set = True
        elif color == "orange" and not self.orange_loading_slot_set:
            if waitpoints:
                random_waitpoint = self.random.choice(waitpoints)
                random_waitpoint.color = "orange"
                self.orange_loading_slot_set = True
        elif color == "purple" and not self.purple_loading_slot_set:
            if waitpoints:
                random_waitpoint = self.random.choice(waitpoints)
                random_waitpoint.color = "purple"
                self.purple_loading_slot_set = True

    def reset_loading_slot(self, color):
        for agent in self.schedule.agents:
            if isinstance(agent, WaitPoint) and agent.color == color:
                agent.color = "yellow"
                break
        if color == "red":
            self.red_loading_slot_set = False
        elif color == "orange":
            self.orange_loading_slot_set = False
        elif color == "purple":
            self.purple_loading_slot_set = False
        self.randomize_loading_slot(color)

    def get_total_time(self):
        return sum(forklift.total_time for forklift in self.forklifts)

    def format_time(self, seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

        if not self.red_loading_slot_set:
            self.randomize_loading_slot("red")
        if not self.orange_loading_slot_set:
            self.randomize_loading_slot("orange")
        if not self.purple_loading_slot_set:
            self.randomize_loading_slot("purple")

        total_time = self.get_total_time()
        formatted_time = self.format_time(total_time)
        energy_consumption_forklifts = [forklift.calculate_energy_consumption() for forklift in self.forklifts]
        total_energy_consumption = sum(energy_consumption_forklifts)

        print(f"Total Time: {formatted_time}")
        print(f"Total Energy Consumption: {total_energy_consumption:.2f} kWh")
        for i, energy in enumerate(energy_consumption_forklifts, start=1):
            print(f"Energy Consumption Forklift {i}: {energy:.2f} kWh")

        combined_unloading_cycles = sum(forklift.unloading_cycles for forklift in self.forklifts)

        if (self.max_time_seconds is not None and total_time >= self.max_time_seconds) or \
           (self.max_cycles is not None and combined_unloading_cycles >= self.max_cycles):
            self.running = False