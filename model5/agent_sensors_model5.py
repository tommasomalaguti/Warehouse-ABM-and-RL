from mesa import Agent
import heapq
import numpy as np

class Forklift(Agent):
    def __init__(self, unique_id, model, empty_speed, loaded_speed, load_time, unload_time):
        super().__init__(unique_id, model)
        self.path_length = 0
        self.energy_consumption = 0
        self.currently_loaded = False
        self.target = None  # Target position to move towards
        self.path = []  # Path to follow
        self.total_distance = 0  # Initialize total distance traveled
        self.unloading_cycles = 0  # Counter for unloading cycles
        self.total_time = 0  # Initialize total time in seconds
        self.speed_empty = empty_speed  # Use sampled empty speed
        self.speed_loaded = loaded_speed  # Use sampled loaded speed
        self.loading_time = load_time  # Use sampled load time
        self.unloading_time = unload_time  # Use sampled unload time
        self.energy_rate = 4.5  # kWh/h according to EN 16796
        self.sample_index = 0  # Initialize sample index
        self.empty_speeds_used = []  # List to store used empty speeds
        self.loaded_speeds_used = []  # List to store used loaded speeds

    def move(self):
        if self.path:
            next_step = self.path.pop(0)
            self.model.grid.move_agent(self, next_step)
            self.path_length += 1
            self.total_distance += 1  # Increment the total distance for each move

            # Determine current speed based on whether the forklift is loaded or not
            if self.currently_loaded:
                cruising_speed = self.speed_loaded
            else:
                cruising_speed = self.speed_empty

            # Calculate time increment based on cruising speed
            time_increment = 1 / cruising_speed
            self.total_time += time_increment
            self.energy_consumption += (time_increment / 3600) * self.energy_rate

            # Check if reached the target
            if self.pos == self.target:
                if self.currently_loaded:
                    print("Reached unloading zone")
                    self.currently_loaded = False
                    self.target = None  # Reset target after unloading
                    self.model.reset_loading_slot()  # Reset the loading slot after unloading
                    self.path = []  # Clear the path
                    self.unloading_cycles += 1  # Increment unloading cycle counter
                    self.total_time += self.unloading_time  # Add unloading time
                    self.energy_consumption += (self.unloading_time / 3600) * self.energy_rate  # Add energy for unloading
                    self.empty_speeds_used.append(self.speed_empty)  # Store used empty speed
                    self.loaded_speeds_used.append(self.speed_loaded)  # Store used loaded speed
                    self.update_parameters()  # Update parameters after unloading cycle
                else:
                    print("Reached loading slot")
                    self.currently_loaded = True
                    self.set_unloading_target()  # Set target to storage zone after picking
                    self.total_time += self.loading_time  # Add loading time
                    self.energy_consumption += (self.loading_time / 3600) * self.energy_rate  # Add energy for loading

    def set_picking_target(self):
        # Find the red cell (picking spot)
        for agent in self.model.schedule.agents:
            if isinstance(agent, WaitPoint) and agent.color == "red":
                self.target = agent.pos
                path = self.a_star_search(self.pos, self.target)
                if path:
                    print(f"Setting path to picking target at {self.target}")
                    self.path = path
                else:
                    print("No valid path to picking target")
                break

    def set_unloading_target(self):
        # Find a storage zone
        storage_zones = [agent for agent in self.model.schedule.agents if isinstance(agent, StorageZone)]
        if storage_zones:
            self.target = self.random.choice(storage_zones).pos
            path = self.a_star_search(self.pos, self.target)
            if path:
                print(f"Setting path to unloading target at {self.target}")
                self.path = path
            else:
                print("No valid path to unloading target")

    def a_star_search(self, start, goal):
        def heuristic(a, b):
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # Chebyshev distance

        def cost(a, b):
            return 1  # Same cost for orthogonal and diagonal moves

        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                break

            for next_pos in self.model.grid.get_neighborhood(current, moore=True, include_center=False):
                if any(isinstance(agent, (NormalWall, StorageWall)) for agent in self.model.grid.get_cell_list_contents(next_pos)):
                    continue  # Skip walls

                new_cost = cost_so_far[current] + cost(current, next_pos)
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(open_list, (priority, next_pos))
                    came_from[next_pos] = current

        if goal not in came_from:
            return None  # No path found

        # Reconstruct path
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def update_parameters(self):
        self.sample_index += 1  # Move to the next sample
        if self.sample_index >= len(self.model.empty_speeds):
            self.sample_index = 0  # Reset index if out of range

        # Update forklift parameters with the next set of samples
        self.speed_empty = self.model.empty_speeds[self.sample_index]
        self.speed_loaded = self.model.loaded_speeds[self.sample_index]
        self.loading_time = self.model.load_times[self.sample_index]
        self.unloading_time = self.model.unload_times[self.sample_index]

    def calculate_energy_consumption(self):
        return self.energy_consumption  # kWh

    def step(self):
        if not self.currently_loaded and not self.target:
            self.set_picking_target()
        elif self.currently_loaded and not self.target:
            self.set_unloading_target()
        self.move()


# Wall and Zone Agents
class NormalWall(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.color = "blue"  # Set default color

class StorageWall(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.color = "black"  # Set default color

class StorageZone(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos

class WaitPoint(Agent):
    def __init__(self, pos, model):
        super().__init__(pos, model)
        self.pos = pos
        self.color = "yellow"  # Set default color
