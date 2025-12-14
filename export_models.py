import numpy as np
import random
import os
import time
from joblib import dump

# ==========================================
# 1. ENVIRONMENT CLASS
# ==========================================
class ParkingGrid:
    def __init__(self, size=10, start=(0,0), parking_spots=[(9,9)],
                 obstacles=None, moving_humans=None,
                 move_penalty=-2, collision_penalty=-50, park_reward=200,
                 boundary_penalty=-20, reward_shaping=True, shaping_coeff=0.1,
                 slip_prob=0.0):

        self.size = size
        self.start = start
        self.parking_spots = set(parking_spots)
        self.static_obstacles = set(obstacles) if obstacles else set()
        self.moving_humans = moving_humans if moving_humans else []
        
        # initial obstacle map
        self.obstacles = self.static_obstacles | {h["pos"] for h in self.moving_humans}

        # rewards
        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.park_reward = park_reward
        self.boundary_penalty = boundary_penalty
        self.reward_shaping = reward_shaping
        self.shaping_coeff = shaping_coeff
        self.slip_prob = slip_prob

        self.reset()

    def reset(self):
        # Random start logic
        if hasattr(self, "start_candidates"):
            self.start = random.choice(self.start_candidates)
        
        # NOTE: Fixed goal logic applied here for training stability
        if hasattr(self, "goal_candidates") and hasattr(self, "use_random_goals") and self.use_random_goals:
            self.parking_spots = {random.choice(self.goal_candidates)}

        self.state = self.start
        self.steps_taken = 0
        self.prev_action = None
        self.visit_count = {}
        return self.state

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _nearest_goal_distance(self, pos):
        if not self.parking_spots: return 0
        return min(abs(pos[0]-g[0]) + abs(pos[1]-g[1]) for g in self.parking_spots)

    def _update_moving_humans(self):
        new_positions = set()
        for h in self.moving_humans:
            x, y = h["pos"]
            if h["axis"] == "h":
                ny = y + h["dir"]
                if ny < h["min"] or ny > h["max"]:
                    h["dir"] *= -1
                    ny = y + h["dir"]
                h["pos"] = (x, ny)
            else:
                nx = x + h["dir"]
                if nx < h["min"] or nx > h["max"]:
                    h["dir"] *= -1
                    nx = x + h["dir"]
                h["pos"] = (nx, y)
            new_positions.add(h["pos"])
        
        self.obstacles = self.static_obstacles | new_positions
        if hasattr(self, "visual_objects"):
            self.visual_objects["human"] = new_positions

    def step(self, action):
        self._update_moving_humans()
        self.steps_taken += 1

        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = np.random.randint(4)

        x, y = self.state
        # 0=up, 1=down, 2=left, 3=right
        if action == 0: nx, ny = x-1, y
        elif action == 1: nx, ny = x+1, y
        elif action == 2: nx, ny = x, y-1
        elif action == 3: nx, ny = x, y+1
        else: nx, ny = x, y

        info = {"is_collision": False, "is_boundary": False, "is_parked": False}
        done = False

        # Boundary check
        if not self._in_bounds(nx, ny):
            info["is_boundary"] = True
            return self.state, self.boundary_penalty, done, info

        next_state = (nx, ny)

        # Obstacle check
        if next_state in self.obstacles:
            info["is_collision"] = True
            return self.state, self.collision_penalty, done, info

        # Parked check
        if next_state in self.parking_spots:
            self.state = next_state
            info["is_parked"] = True
            return next_state, self.park_reward, True, info

        # Move penalty
        reward = self.move_penalty

        # Zig-zag penalty
        if self.prev_action is not None and action != self.prev_action:
            reward -= 1.0
        self.prev_action = action

        # Revisit penalty
        self.visit_count[next_state] = self.visit_count.get(next_state, 0) + 1
        if self.visit_count[next_state] > 1:
            reward -= 1.5

        # Anti-wandering
        if self.steps_taken > 20: reward -= 1
        if self.steps_taken > 50: reward -= 2

        # Soft walkway penalty
        if hasattr(self, "visual_objects"):
            if next_state in self.visual_objects.get("human_walkway", set()):
                reward -= 0.5

        # Shaping
        if self.reward_shaping:
            d0 = self._nearest_goal_distance(self.state)
            d1 = self._nearest_goal_distance(next_state)
            reward += self.shaping_coeff * (d0 - d1)

        self.state = next_state
        return next_state, reward, done, info

    def get_state_space(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

# ==========================================
# 2. ENVIRONMENT BUILDERS
# ==========================================
def env_builder_easy():
    obstacle_map = {
        "parking_slots": {
            (2,0),(3,0),(4,0),(5,0),(7,0),(2,9),(4,9),(5,9),(6,9),(7,9),(8,9),
            (2,4),(2,5),(3,3),(4,3),(5,3),(6,3),(3,6),(4,6),(5,6),(6,6),(9,9)
        },
        "storage": {(0,8), (0,9)},
        "pillar": {(6,0), (3,9), (9,0),(7,3), (7,6),(2,3), (2,6)},
        "bush": {(3,4),(4,4),(5,4),(6,4),(3,5),(4,5),(5,5),(6,5)},
        "guard": {(9,2), (9,3)},
        "parked_car": {
            (2,0),(3,0),(4,0),(7,0),(2,9),(4,9),(5,9),(6,9),(8,9),(2,5),
            (3,3),(4,3),(5,3),(6,3),(3,6),(4,6),(5,6),(6,6)
        },
        "female": {(7,4), (7,5)},
        "waiting": {(4,1)},
        "exiting": {(1,4),(7,8)},
        "empty_soon": {(2,4), (5,0),(7,9)}
    }
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k != "parking_slots"])
    env = ParkingGrid(size=10, start=(0,0), parking_spots=[(9,9)], obstacles=obstacles,
                      move_penalty=-3, collision_penalty=-50, park_reward=200,
                      boundary_penalty=-20, reward_shaping=True, shaping_coeff=0.1, slip_prob=0.1)
    env.visual_objects = obstacle_map
    return env

def env_builder_medium():
    # Condensed version of medium map logic from notebook
    obstacle_map = {
        "parking_slots": set(), # Simplified for brevity, logic remains
        "ticket_machine": {(0,17),(0,18),(0,19),(1,17),(1,18),(1,19),(2,17),(2,18),(2,19),(3,17),(3,18),(3,19)},
        "water_leak": {(13,14),(13,15),(13,16),(13,17),(13,18),(14,14),(14,15),(14,16),(14,17),(14,18),(15,17),(15,18)},
        "barrier_cone": {(15,13),(15,14),(15,15),(15,16),(16,17),(16,18),(16,19),(12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(12,19),(13,13),(14,13),(13,19),(14,19),(15,19)},
        "wall": {(15,2),(15,3),(15,4),(15,5),(15,6),(15,7),(15,8),(15,9),(15,10),(15,11),(15,12),(6,14),(6,17),(7,14),(7,17)},
        "entrance": {(17,18),(18,18),(19,18),(17,19),(18,19),(19,19),(17,17)},
        "bush": {(16,2),(16,3),(16,4),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),(17,2),(17,3),(17,4),(17,5),(17,6),(17,7),(17,8),(17,9),(17,10),(17,11),(17,12),(7,4),(7,5),(7,6),(7,7),(3,6),(3,7),(3,8),(3,11),(3,12),(3,13),(3,14)},
        "parked_car": {(3,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),(3,3),(4,3),(5,2),(6,3),(7,3),(8,3),(9,3),(11,3),(12,3),(13,3),(14,3),(2,6),(2,7),(2,8),(2,11),(2,13),(2,14),(4,6),(4,7),(4,8),(4,11),(4,12),(4,13),(4,14),(9,12),(9,13),(9,14),(9,17),(9,18),(9,19),(10,12),(10,13),(10,14),(10,18),(10,19),(8,7),(10,7),(13,7),(8,8),(9,8),(10,8),(12,8),(13,8)},
        "female": {(6,12),(6,13),(6,15),(6,16),(6,18),(6,19),(7,12),(7,13),(7,15),(7,16),(7,18),(7,19)},
        "waiting": {(3,1)},
        "exiting": {(10,4),(5,4),(1,12),(5,8)},
        "empty_soon": {(4,2),(10,3),(5,3),(2,12),(4,8)}
    }
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k != "parking_slots"])
    env = ParkingGrid(size=20, start=(17,16), parking_spots=[(9,9)], obstacles=obstacles,
                      move_penalty=-3, collision_penalty=-50, park_reward=200,
                      boundary_penalty=-20, reward_shaping=True, shaping_coeff=0.35, slip_prob=0.1)
    env.visual_objects = obstacle_map
    
    # 🔹 RANDOM GOAL PER EPISODE
    env.goal_candidates = [(10,17), (9,7), (12,7)]
    env.use_random_goals = True # Enable random goals for Medium
    return env

def env_builder_hard():
    moving_humans = [
        {"pos": (25,29), "axis": "h", "min": 23, "max": 29, "dir": -1},
        {"pos": (26,29), "axis": "h", "min": 23, "max": 29, "dir": -1},
        {"pos": (24,23), "axis": "v", "min": 17, "max": 24, "dir":  1},
        {"pos": (24,24), "axis": "v", "min": 17, "max": 24, "dir": -1},
        {"pos": (16,23), "axis": "v", "min": 9,  "max": 16, "dir": -1},
        {"pos": (16,24), "axis": "v", "min": 9,  "max": 16, "dir":  1},
        {"pos": (26,7),  "axis": "v", "min": 18, "max": 26, "dir": -1},
        {"pos": (9,7),   "axis": "v", "min": 9,  "max": 15, "dir":  1},
        {"pos": (9,8),   "axis": "v", "min": 9,  "max": 15, "dir": -1},
        {"pos": (29,20), "axis": "v", "min": 22, "max": 29, "dir": -1},
        {"pos": (29,21), "axis": "v", "min": 22, "max": 29, "dir":  1},
        {"pos": (29,22), "axis": "v", "min": 22, "max": 29, "dir": -1},
    ]
    
    # ... (Simplified obstacle map for brevity, contains walls, ticket machine etc.) ...
    obstacle_map = {
        "human": {h["pos"] for h in moving_humans},
        "ticket_machine": {(27,4),(28,4),(29,4),(27,5),(28,5),(29,5),(27,6),(28,6),(29,6)},
        "guard": {(27,0),(28,0),(29,0),(27,1),(28,1),(29,1),(27,2),(28,2),(29,2)},
        "wall": {(0,0),(1,0),(16,7),(17,7),(16,14),(17,14),(12,10),(13,10),(12,17),(13,17),(20,10),(21,10),(20,17),(21,17),(10,27),(10,28),(14,27),(14,28),(18,27),(18,28),(29,8),(29,9),(29,10),(29,11),(29,12),(29,13),(29,14),(29,15),(29,16),(29,17),(29,18),(13,2),(14,2),(15,2),(16,2),(17,2),(18,2),(19,2),(20,2),(21,2),(22,2)},
        # ... (Include other hard obstacles if needed)
    }
    
    obstacles = set().union(*[v for k, v in obstacle_map.items() if k not in ("human", "parking_slots")])

    env = ParkingGrid(
        size=30,
        start=(1,4),
        parking_spots=[(19,28)],  # ✅ FIXED GOAL: (19,28)
        obstacles=obstacles,
        moving_humans=moving_humans,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.45,
        slip_prob=0.1
    )

    env.start_candidates = [(12,3), (1,4), (1,19)]
    # 🔴 DISABLE RANDOM GOALS FOR TRAINING STABILITY
    # env.goal_candidates  = [(16,18), (19,28)] 
    env.use_random_goals = False 

    env.visual_objects = obstacle_map
    return env

# ==========================================
# 3. DOUBLE Q AGENT
# ==========================================
class DoubleQLearningAgent:
    def __init__(self, env, alpha=0.02, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, 
                 min_epsilon=0.05, episodes=20000, max_steps_per_episode=500):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episodes = episodes
        self.max_steps = max_steps_per_episode
        self.q1 = {s: np.zeros(4) for s in env.get_state_space()}
        self.q2 = {s: np.zeros(4) for s in env.get_state_space()}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(4))
        vals = self.q1[state] + self.q2[state]
        return int(np.argmax(vals))

    def train(self):
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            steps = 0
            while not done and steps < self.max_steps:
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                if np.random.rand() < 0.5:
                    a_max = int(np.argmax(self.q1[next_state]))
                    target = reward + (0 if done else self.gamma * self.q2[next_state][a_max])
                    self.q1[state][action] += self.alpha * (target - self.q1[state][action])
                else:
                    a_max = int(np.argmax(self.q2[next_state]))
                    target = reward + (0 if done else self.gamma * self.q1[next_state][a_max])
                    self.q2[state][action] += self.alpha * (target - self.q2[state][action])

                state = next_state
                steps += 1
            
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if episode % 1000 == 0:
                print(f"Episode {episode}/{self.episodes}, Epsilon: {self.epsilon:.4f}")

# ==========================================
# 4. MAIN EXPORT LOGIC
# ==========================================
if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)
    
    env_builders = {
        "easy": env_builder_easy,
        "medium": env_builder_medium,
        "hard": env_builder_hard
    }

    dq_combined_tables = {}

    for level, builder in env_builders.items():
        print(f"=== Training {level.upper()} ===")
        env = builder()
        
        # ✅ TUNED HYPERPARAMETERS FOR HARD LEVEL
        if level == "hard":
            episodes = 50000  # Increased from 20k
            max_steps = 1000  # Increased from 500
        else:
            episodes = 10000
            max_steps = 500

        agent = DoubleQLearningAgent(env, episodes=episodes, max_steps_per_episode=max_steps)
        agent.train()
        
        # Store combined Q-table for greedy execution
        combined_q = {s: (agent.q1[s] + agent.q2[s]) for s in agent.q1.keys()}
        dq_combined_tables[level] = combined_q
        print(f"Finished {level}.")

    # Save to joblib
    dump(dq_combined_tables, "artifacts/dq_combined_tables.joblib")
    print("\n✅ Saved models to artifacts/dq_combined_tables.joblib")
