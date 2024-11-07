import math
import heapq
import numpy as np
from mesa import Agent

class Forklift(Agent):
    def __init__(self, unique_id, model, empty_speed, loaded_speed, load_time, unload_time):
        super().__init__(unique_id, model)
        self.path_length = 0
        self.energy_consumption = 0
        self.currently_loaded = False
        self.target = None  # Target position to move towards
        self.path = []  # Path to follow
        self.total_distance = 0  # Initialize total distance traveled
        self.distance_since_loading = 0  # Initialize distance since last loading
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
        self.load_weight = 0  # Initialize load weight
        self.current_accident_probability = 0.0  # Initialize with zero probability
        self.accidents_count = 0  # Initialize accident counter
        self.accident_probabilities = []  # Initialize list to store accident probabilities
        self.loading_times_used = []
        self.unloading_times_used = []
        self.cycle_times_used = []
        self.weights_lost = []  # List to store the weights lost in accidents
        self.distances_at_accidents = []  # List to store the distances since loading at which accidents occurred
        self.speeds_at_accidents = []  # List to store speeds during accidents
        self.times_at_accidents = []  # List to store times during accidents
        self.random_numbers_at_accidents = []  # List to store random numbers at accidents

    def accident_probability_function(self, speed, weight, distance_traveled, load_time, unload_time):
        """Calculate the probability of an accident based on speed, weight, distance traveled, load time, and unload time."""
        alpha = 0  # Base risk factor
        beta_speed = 0.5  # Increased coefficient for speed
        beta_weight = 0.003  # Coefficient for weight (same as before)
        beta_distance = 0.005  # Coefficient for distance traveled (same as before)
        beta_load_time = -0.04  # Increased coefficient for loading time (decreasing loading time increases risk)
        beta_unload_time = -0.04  # Coefficient for unloading time (decreasing unloading time increases risk)

        # Logistic function to calculate probability
        z = (alpha +
             beta_speed * speed +
             beta_weight * weight +
             beta_distance * distance_traveled +
             beta_load_time * load_time +
             beta_unload_time * unload_time)
        probability = 1 / (1 + np.exp(-z))
        return probability
    
    def move(self):
        if self.path:
            next_step = self.path.pop(0)
            print(f"Moving to next step: {next_step}")
            self.model.grid.move_agent(self, next_step)
            self.path_length += 1
            self.total_distance += 1
            self.distance_since_loading += 1

            # Determine current speed based on whether the forklift is loaded or not
            if self.currently_loaded:
                cruising_speed = self.speed_loaded
                print(f"Forklift is loaded. Moving at loaded speed: {cruising_speed} m/s")

                # Calculate time increment based on cruising speed
                time_increment = 1 / cruising_speed
                self.total_time += time_increment
                self.energy_consumption += (time_increment / 3600) * self.energy_rate

                # Calculate accident probability at every step
                accident_prob = self.accident_probability_function(
                    cruising_speed, self.load_weight, self.distance_since_loading, self.loading_time, self.unloading_time
                )
                self.current_accident_probability = accident_prob  # Update the current accident probability
                print(f"Accident Probability: {accident_prob:.4f}")

                # Generate random number only after two steps since loading
                if self.distance_since_loading == 2:
                    mean = 0.5  # Mean for normal distribution
                    std_dev = 0.1  # Standard deviation for normal distribution
                    random_number = np.random.normal(mean, std_dev)
                    random_number = np.clip(random_number, 0.1, 1.0)  # Ensuring the number stays within the 0.1 to 1.0 range
                    print(f"Random number generated: {random_number:.4f}")
                    if random_number < accident_prob:
                        print("Accident occurred! Load lost.")
                        self.accidents_count += 1
                        self.accident_probabilities.append(self.current_accident_probability)
                        self.weights_lost.append(self.load_weight)
                        self.distances_at_accidents.append(self.distance_since_loading)
                        self.speeds_at_accidents.append((self.speed_empty, self.speed_loaded))
                        self.times_at_accidents.append((self.loading_time, self.unloading_time))
                        self.random_numbers_at_accidents.append(random_number)
                        self.update_parameters()

                        self.model.reset_loading_slot()
                        self.load_weight = 0
                        self.currently_loaded = False
                        self.distance_since_loading = 0
                        self.path = []
                        self.target = None
                        print("Forklift is now unloaded after accident.")

                        # Ensure a new picking target is set
                        self.set_picking_target()
                        if not self.target:
                            print("Warning: No new red waypoint found!")

                        return  # Exit the move function early
                else:
                    print(f"Step {self.distance_since_loading}: Random number will be generated after step 2")

            else:
                cruising_speed = self.speed_empty
                print(f"Forklift is empty. Moving at empty speed: {cruising_speed} m/s")

                time_increment = 1 / cruising_speed
                self.total_time += time_increment
                self.energy_consumption += (time_increment / 3600) * self.energy_rate

            # Check if reached the target
            if self.pos == self.target:
                if self.currently_loaded:
                    print("Reached unloading zone.")
                    print(f"Final Accident Probability before unload: {self.current_accident_probability:.4f}")
                    self.currently_loaded = False
                    self.load_weight = 0
                    self.current_accident_probability = 0.0
                    self.target = None
                    self.model.reset_loading_slot()
                    self.path = []
                    self.unloading_cycles += 1
                    self.total_time += self.unloading_time
                    self.energy_consumption += (self.unloading_time / 3600) * self.energy_rate
                    self.empty_speeds_used.append(self.speed_empty)
                    self.loaded_speeds_used.append(self.speed_loaded)
                    self.loading_times_used.append(self.loading_time)
                    self.unloading_times_used.append(self.unloading_time)
                    self.cycle_times_used.append(self.loading_time + self.unloading_time)
                    self.update_parameters()

                    print(f"Accident Probability reset after unloading: {self.current_accident_probability:.4f}")

                    # Ensure a new picking target is set
                    self.set_picking_target()
                    if not self.target:
                        print("Warning: No new red waypoint found!")

                else:
                    print("Reached loading slot")
                    self.currently_loaded = True
                    self.load_weight = np.random.uniform(500, 3000)
                    print(f"Loaded Weight: {self.load_weight:.2f} kg")
                    self.set_unloading_target()
                    self.total_time += self.loading_time
                    self.energy_consumption += (self.loading_time / 3600) * self.energy_rate
                    self.distance_since_loading = 0
                    self.current_accident_probability = self.accident_probability_function(
                        speed=self.speed_loaded, weight=self.load_weight, distance_traveled=0, load_time=self.loading_time, unload_time=self.unloading_time
                    )
                    print(f"Initial Accident Probability after loading: {self.current_accident_probability:.4f}")
        else:
            print("No path to follow, checking if a new target is needed.")
            if not self.currently_loaded:
                self.set_picking_target()
            elif self.currently_loaded:
                self.set_unloading_target()


    def set_picking_target(self):
        status = "loaded" if self.currently_loaded else "unloaded"
        print(f"Forklift is currently at position {self.pos} and is {status}.")
        print("Attempting to set a picking target...")

        # Check for available red waitpoints
        available_waitpoints = []
        for agent in self.model.schedule.agents:
            if isinstance(agent, WaitPoint) and agent.color == "red":
                available_waitpoints.append(agent)
                print(f"Found red waitpoint at position {agent.pos}")

        if available_waitpoints:
            self.target = available_waitpoints[0].pos
            if self.target == self.pos:
                # Forklift is already at the target position
                print(f"Forklift is already at the target waitpoint at position {self.target}.")
                if not self.currently_loaded:
                    # Perform the loading operation
                    print("Forklift is now loading cargo.")
                    self.currently_loaded = True
                    self.load_weight = np.random.uniform(500, 3000)  # Assign a random weight for the load
                    print(f"Loaded Weight: {self.load_weight:.2f} kg")  # Print the loaded weight
                    self.set_unloading_target()  # Set target to storage zone after picking
                else:
                    print("Forklift is already loaded and ready to move to unloading zone.")
            else:
                path = self.a_star_search(self.pos, self.target)
                if path:
                    print(f"Setting path to picking target at {self.target}")
                    self.path = path
                else:
                    print(f"No valid path to picking target at {self.target} from current position {self.pos}.")
                    self.target = None
        else:
            print("Warning: No red waitpoint found or set!")
            self.target = None



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
