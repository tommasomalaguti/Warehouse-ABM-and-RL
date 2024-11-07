from mesa import Agent
import heapq
import numpy as np

class Forklift2(Agent):
    def __init__(self, unique_id, model, empty_speed, loaded_speed, load_time, unload_time, searching_speed):
        super().__init__(unique_id, model)
        self.path_length = 0
        self.energy_consumption = 0
        self.currently_loaded = False
        self.target = None  # Target position to move towards
        self.path = []  # Path to follow
        self.visited_waitpoints = []  # Track visited waitpoints
        self.previous_waitpoint = None  # Track the previous waitpoint
        self.waitpoints = [(7, 19), (6, 19), (5, 19), (4, 19), (3, 19), (2, 19), (1, 19), (0, 19),
                           (7, 21), (6, 21), (5, 21), (4, 21), (3, 21), (2, 21), (1, 21), (0, 21),
                           (7, 23), (6, 23), (5, 23), (4, 23), (3, 23), (2, 23), (1, 23), (0, 23),
                           (7, 25), (6, 25), (5, 25), (4, 25), (3, 25), (2, 25), (1, 25), (0, 25),
                           (7, 27), (6, 27), (5, 27), (4, 27), (3, 27), (2, 27), (1, 27), (0, 27),
                           (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24),
                           (16, 25), (16, 26), (16, 27), (16, 28), (16, 29), (4, 15), (3, 15), 
                           (2, 15), (1, 15), (0, 15), (4, 13), (3, 13), (2, 13), (1, 13), 
                           (0, 13), (4, 11), (3, 11), (2, 11), (1, 11), (0, 11), (4, 9), 
                           (3, 9), (2, 9), (1, 9), (0, 9), (4, 7), (3, 7), (2, 7), (1, 7), 
                           (0, 7), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5), (4, 3), (3, 3), 
                           (2, 3), (1, 3), (0, 3), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1), 
                           (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (21, 5), 
                           (21, 6), (21, 7), (21, 8), (21, 9), (21, 10), (23, 5), (23, 6), 
                           (23, 7), (23, 8), (23, 9), (23, 10), (25, 5), (25, 6), (25, 7), 
                           (25, 8), (25, 9), (25, 10), (27, 5), (27, 6), (27, 7), (27, 8), 
                           (27, 9), (27, 10), (28, 2), (27, 2), (26, 2), (25, 2), (24, 2), 
                           (23, 2), (22, 2), (21, 2), (20, 2), (19, 2), (23, 22), (22, 22), 
                           (21, 22), (20, 22), (19, 22), (23, 24), (22, 24), (21, 24), (20, 24), 
                           (19, 24), (23, 26), (22, 26), (21, 26), (20, 26), (19, 26), (23, 28), 
                           (22, 28), (21, 28), (20, 28), (19, 28), (25, 28), (26, 28), (27, 28), 
                           (28, 28), (25, 26), (26, 26), (27, 26), (28, 26), (25, 24), (26, 24), 
                           (27, 24), (28, 24), (25, 22), (26, 22), (27, 22), (28, 22)]
        self.current_waitpoint_index = 0
        self.total_distance = 0  # Initialize total distance traveled
        self.unloading_cycles = 0  # Counter for unloading cycles
        self.total_time = 0  # Initialize total time in seconds
        self.speed_empty = empty_speed  # Use sampled empty speed
        self.speed_loaded = loaded_speed  # Use sampled loaded speed
        self.loading_time = load_time  # Use sampled load time
        self.unloading_time = unload_time  # Use sampled unload time
        self.searching_speed = searching_speed  # Use sampled searching speed
        self.energy_rate = 4.5  # kWh/h according to EN 16796
        self.sample_index = 0  # Initialize sample index
        self.empty_speeds_used = []  # List to store used empty speeds
        self.loaded_speeds_used = []  # List to store used loaded speeds
        self.speed_searching_used = []  # List to store used searching speeds

    def move(self):
        if self.path:
            next_step = self.path.pop(0)

            # Move to the next step first
            self.model.grid.move_agent(self, next_step)
            self.path_length += 1  # Each step is 1 meter
            self.total_distance += 1  # Increment the total distance by 1 meter for each move

            # After moving, update the color of the previous waitpoint to grey
            if not self.currently_loaded and self.previous_waitpoint and self.previous_waitpoint != self.pos:
                for agent in self.model.grid.get_cell_list_contents(self.previous_waitpoint):
                    if isinstance(agent, WaitPoint):
                        agent.color = "grey"
                        self.visited_waitpoints.append(agent)

            # Set the current position as the previous waitpoint for the next move
            if self.pos in self.waitpoints:
                self.previous_waitpoint = self.pos

            # Determine speed based on the location (if it's a waitpoint or not)
            speed = self.speed_empty  # Default to empty speed
            if self.pos in self.waitpoints:
                for agent in self.model.grid.get_cell_list_contents(self.pos):
                    if isinstance(agent, WaitPoint) and agent.color == "grey":
                        speed = self.speed_empty
                        break
                else:
                    speed = self.searching_speed
                    self.speed_searching_used.append(speed)  # Store used searching speed
            elif self.currently_loaded:
                speed = self.speed_loaded

            # Calculate time and energy consumption based on the determined speed
            time_increment = 1 / speed
            self.total_time += time_increment
            self.energy_consumption += (time_increment / 3600) * self.energy_rate

            # Check if reached the target
            if self.pos == self.target:
                if self.currently_loaded:
                    self.currently_loaded = False
                    self.target = None  # Reset target after unloading
                    self.model.reset_loading_slot()  # Reset the loading slot after unloading
                    self.path = []  # Clear the path
                    self.current_waitpoint_index = 0  # Reset waitpoint index
                    self.unloading_cycles += 1  # Increment unloading cycle counter
                    self.total_time += self.unloading_time  # Add unloading time
                    self.energy_consumption += (self.unloading_time / 3600) * self.energy_rate  # Add energy for unloading
                    self.empty_speeds_used.append(self.speed_empty)  # Store used empty speed
                    self.loaded_speeds_used.append(self.speed_loaded)  # Store used loaded speed
                    self.update_parameters()  # Update parameters after unloading cycle
                    self.reset_grey_waitpoints()  # Reset all grey waitpoints to yellow
                    self.previous_waitpoint = None  # Reset the previous waitpoint
                else:
                    # Check if the current waitpoint is red
                    for agent in self.model.grid.get_cell_list_contents(self.pos):
                        if isinstance(agent, WaitPoint) and agent.color == "red":
                            self.currently_loaded = True
                            self.set_unloading_target()  # Set target to storage zone after picking
                            self.total_time += self.loading_time  # Add loading time
                            self.energy_consumption += (self.loading_time / 3600) * self.energy_rate  # Add energy for loading
                            self.reset_grey_waitpoints()  # Reset all grey waitpoints to yellow
                            return  # Exit after setting the unloading target
                    self.set_picking_target()  # Continue to the next waitpoint

    def set_picking_target(self):
        # Follow specific path through all waitpoints
        if self.current_waitpoint_index < len(self.waitpoints):
            self.target = self.waitpoints[self.current_waitpoint_index]
            path = self.a_star_search(self.pos, self.target)
            if path:
                print(f"Setting path to waitpoint at {self.target}")
                self.path = path
                self.current_waitpoint_index += 1
            else:
                print("No valid path to waitpoint")
        else:
            # Check for red waitpoint and go to storage zone
            for agent in self.model.schedule.agents:
                if isinstance(agent, WaitPoint) and agent.color == "red":
                    self.target = agent.pos
                    path = self.a_star_search(self.pos, self.target)
                    if path:
                        print(f"Found red waitpoint at {self.target}, setting path")
                        self.path = path
                        return

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
        self.searching_speed = self.model.searching_speed[self.sample_index]

        # Print or log the updated speeds for debugging purposes
        print(f"Updated speeds - Empty: {self.speed_empty}, Loaded: {self.speed_loaded}, Searching: {self.searching_speed}")


    def reset_grey_waitpoints(self):
        for waitpoint in self.visited_waitpoints:
            waitpoint.color = "yellow"
        self.visited_waitpoints = []

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