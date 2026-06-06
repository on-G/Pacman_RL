import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from collections import deque

class GridWorldEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.width = 25
        self.height = 22
        self.window_width = 1250
        self.window_height = 1100
        self.render_mode = render_mode

        self.observation_space = spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), high=np.array([self.width-1, self.height-1,1,1,1,1,25,21]), dtype=np.int32)

        self.action_space = spaces.Discrete(4)

        self.goal = np.array([24,0])

        self.walls = [(12,0),(1,1),(2,1),(3,1),(4,1),(6,1),(7,1),(8,1),(9,1),(10,1),(12,1),(14,1),(15,1),(16,1),(17,1),
                      (18,1),(20,1),(21,1),(22,1),(23,1),(1,2),(2,2),(3,2),(4,2),(6,2),(7,2),(8,2),(9,2),(10,2),(12,2),
                      (14,2),(15,2),(16,2),(17,2),(18,2),(20,2),(21,2),(22,2),(23,2),(1,3),(2,3),(3,3),(4,3),(6,3),(7,3),
                      (8,3),(9,3),(10,3),(12,3),(14,3),(15,3),(16,3),(17,3),(18,3),(20,3),(21,3),(22,3),(23,3),(1,5),(2,5),
                      (3,5),(4,5),(6,5),(7,5),(9,5),(10,5),(11,5),(12,5),(13,5),(14,5),(15,5),(17,5),(18,5),(20,5),(21,5),
                      (22,5),(23,5),(6,6),(7,6),(12,6),(17,6),(18,6),(0,7),(1,7),(2,7),(3,7),(4,7),(6,7),(7,7),(8,7),(9,7),
                      (12,7),(15,7),(16,7),(17,7),(18,7),(20,7),(21,7),(22,7),(23,7),(24,7),(4,8),(6,8),(7,8),(8,8),(9,8),
                      (15,8),(16,8),(17,8),(18,8),(20,8),(4,9),(6,9),(7,9),(17,9),(18,9),(20,9),(4,10),(6,10),(7,10),(17,10),
                      (18,10),(20,10),(4,11),(20,11),(4,12),(6,12),(7,12),(17,12),(18,12),(20,12),(4,13),(6,13),(7,13),(17,13),
                      (18,13),(20,13),(0,14),(1,14),(2,14),(3,14),(4,14),(6,14),(7,14),(9,14),(10,14),(11,14),(12,14),(13,14),
                      (14,14),(15,14),(17,14),(18,14),(20,14),(21,14),(22,14),(23,14),(24,14),(12,15),(1,16),(2,16),(3,16),
                      (4,16),(6,16),(7,16),(8,16),(9,16),(10,16),(12,16),(14,16),(15,16),(16,16),(17,16),(18,16),(20,16),
                      (21,16),(22,16),(23,16),(4,17),(20,17),(0,18),(1,18),(2,18),(4,18),(6,18),(9,18),(10,18),(11,18),
                      (12,18),(13,18),(14,18),(15,18),(18,18),(20,18),(22,18),(23,18),(24,18),(6,19),(12,19),(18,19),(1,20),
                      (2,20),(3,20),(4,20),(5,20),(6,20),(7,20),(8,20),(9,20),(12,20),(15,20),(16,20),(17,20),(18,20),(19,20),
                      (20,20),(21,20),(22,20),(23,20)]

        self.max_steps = 400

        self.window = None

        self.clock = None

        self.path = []
        self.ghost_path = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = np.array([0,21])
        self.ghost_pos = np.array([20,21])

        self.steps = 0

        self.path = [tuple(self.agent_pos)]

        return self.get_obs(), {}

    def step(self, action):
        self.steps += 1

        new_position = self.agent_pos.copy()

        if action == 0:
            new_position[1] -= 1

        if action == 1:
            new_position[1] += 1

        if action == 2:
            new_position[0] -= 1

        if action == 3:
            new_position[0] += 1

        new_position = np.clip(new_position, 0, [self.width-1, self.height-1])

        candidate_position = (int(new_position[0]), int(new_position[1]))

        if candidate_position not in self.walls:
            self.agent_pos = new_position.copy()

        self.path.append((int(self.agent_pos[0]), int(self.agent_pos[1])))

        #distance = np.linalg.norm(self.agent_pos - self.goal)
        #reward = -0.1 * distance

        self.move_ghost()

        caught = np.array_equal(self.ghost_pos, self.agent_pos)
        reached_goal = np.array_equal(self.agent_pos, self.goal)

        terminated = (caught or reached_goal)


        if reached_goal:
            reward = 10
        elif caught:
            reward = -10
        else:
            reward = -0.01
        
        truncated = self.steps >= self.max_steps

        print("Current: ", self.agent_pos, "Action: ", action, "Candidate: ", candidate_position, "Ghost Distance: ", len(self.ghost_path))
        if candidate_position in self.walls:
            print("Wall collision")

        return (self.get_obs(), reward, terminated, truncated, {})
    
    def render(self):

        if self.window is None:
            pygame.init()

            self.window = pygame.display.set_mode((self.window_width, self.window_height))

            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))

        canvas.fill((255, 255, 255))

        cell_size = self.window_height / self.height

        #Draw path
        for position in self.path:
            pygame.draw.rect(canvas, (200, 220, 255), pygame.Rect(position[0]*cell_size, position[1]*cell_size, cell_size, cell_size))

         #Draw ghost path
        #for position in self.ghost_path:
            #pygame.draw.rect(canvas, (255, 220, 200), pygame.Rect(position[0]*cell_size, position[1]*cell_size, cell_size, cell_size))

        #Draw walls
        for wall in self.walls:
            pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(wall[0]*cell_size, wall[1]*cell_size, cell_size, cell_size))

        #Draw goal
        pygame.draw.rect(canvas, (0, 255, 0), pygame.Rect(self.goal[0]*cell_size, self.goal[1]*cell_size, cell_size, cell_size))

        #Draw ghost
        pygame.draw.circle(canvas, (255, 0, 0), (int(self.ghost_pos[0]*cell_size + cell_size/2), int(self.ghost_pos[1]*cell_size + cell_size/2)), int(cell_size/3))

        #Draw agent
        pygame.draw.circle(canvas, (0, 0, 255), (int(self.agent_pos[0]*cell_size + cell_size/2), int(self.agent_pos[1]*cell_size + cell_size/2)), int(cell_size / 3))

        #Draw grid lines
        for x in range(self.window_width+1):
            pygame.draw.line(canvas, (180, 180, 180), (x*cell_size, 0), (x*cell_size, self.window_height))

        for y in range(self.window_height+1):
            pygame.draw.line(canvas, (180, 180, 180), (0, y*cell_size), (self.window_width, y*cell_size))

        self.window.blit(canvas, (0,0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()

    def is_wall(self, x, y):
        if(x < 0 or x >= self.width or y < 0 or y >= self.height):
            return 1

        return int((x,y) in self.walls)
    
    def get_obs(self):
        x = int(self.agent_pos[0])
        y = int(self.agent_pos[1])

        dx = self.agent_pos[0] - self.ghost_pos[0]
        dy = self.agent_pos[1] - self.ghost_pos[1]

        return np.array([x,y, self.is_wall(x,y+1),self.is_wall(x,y-1),self.is_wall(x+1,y),self.is_wall(x-1,y), dx, dy])

    def get_neighbors(self, position):
        x, y = position

        neighbors = [(x+1,y),(x-1,y),(x,y-1),(x,y+1)]
        valid = []

        for nx, ny in neighbors:
            if (0 <= nx < self.width and 0 <= ny < self.height and (nx,ny) not in self.walls):
                valid.append((nx,ny))

        return valid
    
    def bfs_path(self, start, goal):
        queue = deque()
        queue.append(start)
        visited = set()
        visited.add(start)
        parent = {}

        while queue:
            current = queue.popleft()

            if current == goal:
                path = []

                while current != start:
                    path.append(current)
                    current = parent[current]

                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return []
    
    def move_ghost(self):
        start = (int(self.ghost_pos[0]),int(self.ghost_pos[1]))
        goal = (int(self.agent_pos[0]),int(self.agent_pos[1]))

        self.ghost_path = self.bfs_path(start, goal)

        if len(self.ghost_path) > 0:
            next_position = self.ghost_path[0]
            self.ghost_pos = np.array(next_position)