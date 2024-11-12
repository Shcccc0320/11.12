import numpy as np
import gym
from gym import spaces
import networkx as nx
import shap

class V2XEnvironment(gym.Env):
    def __init__(self):
        super(V2XEnvironment, self).__init__()

        # Initialize grid road network
        self.G = nx.Graph()
        # Add nodes and positions for intersections
        self.G.add_nodes_from([
            ('A', {'position': (0, 0)}),  # Intersection nodes
            ('B', {'position': (0, 3)}),
            ('C', {'position': (3, 0)}),
            ('D', {'position': (3, 3)})
        ])
        # Add edges to represent roads
        self.G.add_edges_from([
            ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')
        ])

        # Base station configuration (0 is MBS, 1-4 are SBS)
        self.base_stations = {
            0: {'position': (1.5, 1.5), 'total_bandwidth': 300, 'transmission_power': 120},  # MBS configuration
            1: {'position': (0, 0), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at A
            2: {'position': (0, 3), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at B
            3: {'position': (3, 0), 'total_bandwidth': 100, 'transmission_power': 60},  # SBS at C
            4: {'position': (3, 3), 'total_bandwidth': 100, 'transmission_power': 60}   # SBS at D
        }

        self.noise_power = 1e-13  # Noise power

        self.num_vehicles = 4  # Number of vehicles
        self.num_stations = 5  # Number of base stations

        # Total available bandwidth
        self.total_available_bandwidth = 50e6  # 10 MHz
        # QoS requirements and penalty factors
        self.max_delay = 0.01  # Maximum acceptable delay for URLLC (seconds)
        self.min_rate = 10  # Minimum acceptable data rate for eMBB (Mbps)

        # Normalization constants for state representation
        self.max_position = 3.0  # Maximum coordinate value
        self.max_data_requirement = 80.0  # Maximum data requirement
        self.max_bandwidth = self.total_available_bandwidth  # Use total available bandwidth
        # Time step and step counter
        self.delta_t = 1.0  # Time step duration
        self.step_count = 0

        # Define observation space and action space
        self.observation_space = self._get_observation_space()  # Define the observation space
        self.action_space = self._get_action_space()  # Define the action space

        # Initialize environment state
        self.state = self.reset()  # Initialize environment state

        # Additional variables to store experience data for SHAP analysis
        self.experiences = []
        self.explainer = None

    def _get_observation_space(self):
        # Define the observation space size based on vehicle and base station states
        vehicle_state_size = 2 + 1 + 1 + 3 + 1  # Position (x, y), data requirement, bandwidth allocation, slice type, active flag
        base_station_state_size = self.num_stations * (2 + 1)  # Distance, channel gain, total bandwidth per base station
        total_state_size = self.num_vehicles * (vehicle_state_size + base_station_state_size)
        return spaces.Box(low=0, high=1, shape=(total_state_size,), dtype=np.float32)

    def _get_action_space(self):
        # Define the action space: base station selection and bandwidth allocation for each vehicle
        base_station_selection_space = spaces.MultiDiscrete([self.num_stations for _ in range(self.num_vehicles)])
        bandwidth_allocation_space = spaces.Box(low=0, high=1, shape=(self.num_vehicles,), dtype=np.float32)
        return spaces.Tuple((base_station_selection_space, bandwidth_allocation_space))

    def reset(self):
        # Initialize vehicle positions and states
        routes = [
            ['A', 'B', 'D'],  # Vehicle 0 route
            ['C', 'D', 'B'],  # Vehicle 1 route
            ['B', 'A', 'C'],  # Vehicle 2 route
            ['D', 'C', 'A'],  # Vehicle 3 route
        ]

        self.state = {
            'vehicles': {
                i: {
                    'route': routes[i],
                    'route_index': 0,  # Current node index in the route
                    'current_edge': (routes[i][0], routes[i][1]),  # Current edge (from_node, to_node)
                    'position': self.G.nodes[routes[i][0]]['position'],  # Current position coordinates
                    'distance_on_edge': 0.0,  # Distance moved along the current edge
                    'edge_length': self.calculate_edge_length(routes[i][0], routes[i][1]),  # Length of the current edge
                    'velocity': np.random.uniform(0.01, 0.05),  # Vehicle speed
                    'slice_type': np.random.choice(['URLLC', 'eMBB', 'Both']),
                    'has_arrived': False,
                    'data_requirement': None,
                    'bandwidth_allocation': 0.0,
                    'base_station': None,  # Initially not connected to any base station
                }
                for i in range(self.num_vehicles)
            }
        }

        # Set data requirements
        for vehicle in self.state['vehicles'].values():
            slice_type = vehicle['slice_type']
            if slice_type == 'URLLC':
                vehicle['data_requirement'] = max(np.random.uniform(10, 80), 1e-3)  # Data size (Mbits)
            elif slice_type == 'eMBB':
                vehicle['data_requirement'] = max(np.random.uniform(20, 100), 1e-3)
            elif slice_type == 'Both':
                vehicle['data_requirement'] = max(np.random.uniform(10, 100), 1e-3)
            vehicle['bandwidth_allocation'] = 0.0

        return self.get_state()

    def calculate_edge_length(self, node1, node2):
        pos1 = np.array(self.G.nodes[node1]['position'])
        pos2 = np.array(self.G.nodes[node2]['position'])
        return np.linalg.norm(pos2 - pos1)



    def step(self, action):
        # Execute one time step within the environment
        self.step_count += 1  # Increment step counter

        # Record the current state and action for later analysis
        current_state = self.get_state()
        self.experiences.append((current_state, action))

        # Apply the action, move vehicles, and calculate reward
        self.apply_action(action)
        self.move_vehicles()
        reward = self.calculate_reward()

        # Calculate XAI reward
        xai_reward = self.calculate_xai_reward(action)
        reward += xai_reward + self.calculate_reward()

        # Get the next state
        next_state = self.get_state()

        done = self.check_if_done()

        info = {}
        return next_state, reward, done, info

    def calculate_xai_reward(self, action):
        # Calculate the XAI reward based on the entropy of the action
        entropy = self.calculate_entropy(action)
        return 1 / (entropy + 1e-6)  # XAI reward

    def calculate_entropy(self, action):
        # Calculate the entropy of the action to measure uncertainty
        action_probabilities = np.ones(len(action)) / len(action)  # Assume equal probability for each action
        entropy = -np.sum(action_probabilities * np.log(action_probabilities + 1e-6))
        return entropy

    def explain_action(self, state, action):
        # Use SHAP to explain the action taken given the state
        if self.explainer:
            shap_values = self.explainer.shap_values(state)
            shap.summary_plot(shap_values, state)

    def apply_action(self, action):
        # Apply the action to the environment, updating vehicle states
        base_station_selection, bandwidth_allocations = action
        for i in range(self.num_vehicles):
            vehicle = self.state['vehicles'][i]
            if vehicle.get('has_arrived', False):
                # Ignore actions for vehicles that have arrived at their destination
                vehicle['base_station'] = None
                vehicle['bandwidth_allocation'] = 0.0
                continue
            base_station_index = base_station_selection[i]  # Get the selected base station for the vehicle
            bandwidth_fraction = np.clip(bandwidth_allocations[i], 0, 1)  # Clip bandwidth allocation between 0 and 1

            vehicle['base_station'] = base_station_index  # Set base station selection
            total_bandwidth = self.base_stations[base_station_index]['total_bandwidth']  # Get total bandwidth of the base station
            vehicle['bandwidth_allocation'] = bandwidth_fraction * total_bandwidth  # Calculate and set bandwidth allocation for the vehicle

    def move_vehicles(self):
        for vehicle_id, vehicle in self.state['vehicles'].items():
            if vehicle.get('has_arrived', False):
                continue

            # 获取当前边的信息
            from_node, to_node = vehicle['current_edge']
            edge_length = vehicle['edge_length']

            # 计算本次移动的距离
            move_distance = vehicle['velocity']

            # 更新在当前边上已经移动的距离
            vehicle['distance_on_edge'] += move_distance

            # print(
            # f"Vehicle {vehicle_id} before moving: position {vehicle['position']}, distance_on_edge {vehicle['distance_on_edge']}, edge_length {edge_length}, route_index {vehicle['route_index']}")

            while vehicle['distance_on_edge'] >= edge_length:
                # 到达当前边的终点，更新到下一个边
                vehicle['distance_on_edge'] -= edge_length
                vehicle['route_index'] += 1

                if vehicle['route_index'] >= len(vehicle['route']) - 1:
                    # 已经到达最终目的地
                    vehicle['has_arrived'] = True
                    vehicle['position'] = self.G.nodes[vehicle['route'][-1]]['position']
                    vehicle['current_edge'] = None
                    vehicle['edge_length'] = 0.0
                    vehicle['distance_on_edge'] = 0.0
                    # print(f"Vehicle {vehicle_id} has arrived at the destination at step {self.step_count}.")
                    break
                else:
                    # 更新当前边的信息
                    from_node = vehicle['route'][vehicle['route_index']]
                    to_node = vehicle['route'][vehicle['route_index'] + 1]
                    vehicle['current_edge'] = (from_node, to_node)
                    edge_length = self.calculate_edge_length(from_node, to_node)
                    vehicle['edge_length'] = edge_length
                    # print(
                    # f"Vehicle {vehicle_id} moves to next edge from {from_node} to {to_node} at step {self.step_count}.")

            if not vehicle.get('has_arrived', False):
                # 根据在当前边上的位置计算当前位置
                from_pos = np.array(self.G.nodes[from_node]['position'])
                to_pos = np.array(self.G.nodes[to_node]['position'])
                ratio = vehicle['distance_on_edge'] / edge_length if edge_length > 0 else 0.0
                ratio = np.clip(ratio, 0, 1)
                new_position = from_pos + (to_pos - from_pos) * ratio
                vehicle['position'] = new_position.tolist()
                # print(
                # f"Vehicle {vehicle_id} after moving: position {vehicle['position']}, distance_on_edge {vehicle['distance_on_edge']}, route_index {vehicle['route_index']}")

    def get_state(self):
        # Get the current state representation of the environment
        state = []

        for vehicle in self.state.get('vehicles', {}).values():
            # Active flag to indicate if the vehicle is still in transit
            active_flag = 0.0 if vehicle.get('has_arrived', False) else 1.0  # Set to 0 if the vehicle has arrived
            # Normalize vehicle position, data requirement, and bandwidth allocation
            vehicle_state = [
                (vehicle['position'][0] / self.max_position) * active_flag,  # Normalized x-coordinate
                (vehicle['position'][1] / self.max_position) * active_flag,  # Normalized y-coordinate
                (vehicle['data_requirement'] / self.max_data_requirement) * active_flag,  # Normalized data requirement
                (vehicle['bandwidth_allocation'] / self.max_bandwidth) * active_flag,  # Normalized bandwidth allocation
                # One-hot encoding of slice type, multiplied by active flag
                (1.0 if vehicle['slice_type'] == 'URLLC' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'eMBB' else 0.0) * active_flag,
                (1.0 if vehicle['slice_type'] == 'Both' else 0.0) * active_flag,
                active_flag  # Add active flag to state
            ]

            # Add distance, channel gain, and total bandwidth for each base station
            for base_station_id, base_station in self.base_stations.items():
                distance = np.linalg.norm(
                    np.array(vehicle['position']) - np.array(base_station['position'])) / 3.0  # Normalized distance
                channel_gain = self.calculate_channel_gain(vehicle['position'], base_station['position'])  # Calculate channel gain
                base_station_state = [
                    distance * active_flag,  # Normalized distance
                    channel_gain * active_flag,  # Channel gain
                    (base_station['total_bandwidth'] / self.max_bandwidth) * active_flag,  # Normalized total bandwidth
                ]
                vehicle_state.extend(base_station_state)
            state.extend(vehicle_state)

        state_array = np.array(state, dtype=np.float32)  # Convert state to NumPy array
        return state_array

    def calculate_channel_gain(self, vehicle_position, bs_position):
        # Calculate the channel gain between a vehicle and a base station
        distance = np.linalg.norm(np.array(vehicle_position) - np.array(bs_position))
        distance = max(distance, 1e-3)  # Avoid division by zero
        path_loss_exponent = 2.0  # Path loss exponent
        return (1 / distance) ** path_loss_exponent

    def calculate_reward(self):
        # Calculate the reward based on QoS requirements and vehicle states
        reward = 0.0
        for vehicle_id, vehicle in self.state.get('vehicles', {}).items():
            if vehicle.get('has_arrived', False):
                continue
            base_station_id = vehicle['base_station']
            slice_type = vehicle['slice_type']
            datarate = self.calculate_datarate(vehicle, base_station_id)
            delay = self.calculate_delay(vehicle, base_station_id)
            if slice_type == 'URLLC':
                reward += 2.0 if delay <= 0.01 else -0.1 * delay  # Reward for URLLC based on delay
            elif slice_type == 'eMBB':
                reward += 2.0 if datarate >= 10 else 0.05 * datarate  # Reward for eMBB based on data rate
            elif slice_type == 'Both':
                if delay <= 0.01 and datarate >= 10:
                    reward += 3.0  # Reward if both URLLC and eMBB requirements are met
                else:
                    reward -= 0.5 * delay  # Penalty for delay
                    reward += 0.05 * datarate  # Reward for data rate
        return np.clip(reward, -10, 10)  # Clip the reward to keep it within a reasonable range

    def calculate_datarate(self, vehicle, base_station_id):
        # Calculate the data rate for a vehicle connected to a base station
        W_i_m = vehicle['bandwidth_allocation']
        if W_i_m <= 1e-3 or base_station_id is None:
            return 0.0
        P = self.base_stations[base_station_id]['transmission_power']
        Gt_i_m = self.calculate_channel_gain(vehicle['position'], self.base_stations[base_station_id]['position'])
        sigma2 = self.noise_power
        sinr = (P * Gt_i_m) / (sigma2 + 0.1)  # Signal-to-interference-plus-noise ratio (SINR)
        return W_i_m * np.log2(1 + sinr) / 1e6 if sinr > 0 else 0.0  # Data rate in Mbps

    def calculate_delay(self, vehicle, base_station_id):
        # Calculate the delay for a vehicle based on data requirement and data rate
        data_size = vehicle['data_requirement']
        datarate_mbps = self.calculate_datarate(vehicle, base_station_id)
        return data_size / datarate_mbps if datarate_mbps > 1e-3 else 0.1  # Delay in seconds

    def check_if_done(self):
        all_arrived = all(vehicle.get('has_arrived', False) for vehicle in self.state['vehicles'].values())
        # print(f"Check if done: {all_arrived}")
        return all_arrived

    def render(self, mode='human'):
        # Render the current state of the environment using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for bs_id, bs in self.base_stations.items():
            plt.scatter(bs['position'][0], bs['position'][1], c='red' if bs_id == 0 else 'green', marker='^' if bs_id == 0 else 'v', s=200)
            plt.text(bs['position'][0] + 0.05, bs['position'][1] + 0.05, f'MBS' if bs_id == 0 else f'SBS{bs_id}')
        pos = nx.get_node_attributes(self.G, 'position')
        nx.draw(self.G, pos, node_color='gray', node_size=50, with_labels=True)
        for vehicle in self.state.get('vehicles', {}).values():
            if vehicle.get('has_arrived', False):
                continue
            plt.scatter(vehicle['position'][0], vehicle['position'][1], c='blue', s=100)
        plt.xlim(-1, 4)
        plt.ylim(-1, 4)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('V2X Environment')
        plt.show()
        plt.close()

    def close(self):
        # Close the environment (no specific action needed here)
        pass

