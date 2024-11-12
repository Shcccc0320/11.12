import numpy as np
from V2XEnvironment import V2XEnvironment  # Ensure this matches your file name

# Create environment instance
env = V2XEnvironment()

# Reset environment, get initial state
state = env.reset()

done = False
total_reward = 0
step = 0
render_interval = 7  # Render every 10 steps to avoid overloading the rendering system

while not done:
    # Simple policy: randomly select base station, allocate 50% bandwidth
    base_station_selection = np.random.randint(0, env.num_stations, size=env.num_vehicles)
    bandwidth_allocations = np.full(env.num_vehicles, 0.5)
    action = (base_station_selection, bandwidth_allocations)

    # Execute action, get next state, reward, and done flag
    next_state, reward, done, info = env.step(action)

    total_reward += reward
    step += 1

    # Render environment at specified intervals
    if step % render_interval == 0:
        env.render()

    # Update state
    state = next_state

print(f"Episode finished after {step} steps, total reward: {total_reward}")
