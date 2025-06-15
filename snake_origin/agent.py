# In snake_origin/agent.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point # game.py 不變
from model import Linear_QNet, QTrainer # model.py 已修改
from helper import plot # helper.py 將修改

# 設定 device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device) # 模型實例化後移至 device
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # model 已在 device 上


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # QTrainer.train_step 內部會處理張量的 device 轉換
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss # 返回 loss

    def train_short_memory(self, state, action, reward, next_state, done):
        # QTrainer.train_step 內部會處理張量的 device 轉換
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss # 返回 loss

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device) # 將 state 張量移至 device
            prediction = self.model(state0) # model 已在 device 上
            move = torch.argmax(prediction).item() # .item() 將結果移回 CPU
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    plot_losses = [] # 新增：用於儲存每局遊戲的 loss
    plot_mean_losses = [] # 新增：用於儲存平均 loss
    total_score = 0
    total_loss = 0 # 新增：用於計算總 loss
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    print(f"Training on device: {device}") # 顯示使用的 device

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        # short_term_loss = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # (選擇性：如果需要追蹤每一步的 loss，可以取消註釋並收集)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)


        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            batch_loss = agent.train_long_memory() # 獲取 long term (batch) loss

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record, f'Batch Loss: {batch_loss:.4f}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # 儲存並計算平均 loss (使用來自 train_long_memory 的 batch_loss)
            plot_losses.append(batch_loss) 
            total_loss += batch_loss
            mean_loss_val = total_loss / agent.n_games # 避免與 mean_loss 函數名衝突
            plot_mean_losses.append(mean_loss_val)

            plot(plot_scores, plot_mean_scores, plot_losses, plot_mean_losses) # 傳遞 losses 給 plot 函數


if __name__ == '__main__':
    train()