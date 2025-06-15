import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point # game.py 不變
from model import Linear_QNet, QTrainer # model.py 已修改
#from helper import plot # helper.py 將修改
import json # 新增：用於保存 JSON 檔案
import os # 新增：用於檢查檔案路徑
from datetime import datetime # 新增：用於為檔案名添加時間戳
from multiprocessing import Process, Queue # <-- 新增
from live_plotter import plot_process_target # <-- 新增
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
        #define the points around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        #check direction (only single direction gets 1 at the same time, others remain 0)
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or #going right & collision at right = danger straight
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or #logicly same as danger straight, etc.
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            #like current direction, only single direction gets 1.
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
        self.memory.append((state, action, reward, next_state, done)) 
        #appending memries
        # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        #1 batch per train
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # QTrainer.train_step 內部會處理張量的 device 轉換
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss # 返回 loss

    def train_short_memory(self, state, action, reward, next_state, done):
        #Only this run's memmory
        # QTrainer.train_step 內部會處理張量的 device 轉換
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss # 返回 loss

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games #by increasing the number of games played, the randomness decreasses
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: #move on a random direction
            move = random.randint(0, 2)
            final_move[move] = 1
        else: #based on exprecience, predict the move
            state0 = torch.tensor(state, dtype=torch.float).to(device) # 將 state 張量移至 device
            prediction = self.model(state0) # model 已在 device 上
            move = torch.argmax(prediction).item() # .item() 將結果移回 CPU
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    plot_mean_losses = []
    total_score = 0
    total_loss = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    print(f"Training on device: {device}")

    # ===== 新增：設定 multiprocessing =====
    plot_queue = Queue()
    plot_proc = Process(target=plot_process_target, args=(plot_queue,))
    plot_proc.start()
    # ==================================

    data_to_save = {
        'scores': plot_scores,
        'mean_scores': plot_mean_scores,
        'losses': plot_losses,
        'mean_losses': plot_mean_losses,
        'records_over_time': []
    }

    training_data_folder = './training_data_dynamic'
    if not os.path.exists(training_data_folder):
        os.makedirs(training_data_folder)

    try:
        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                batch_loss = agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    data_to_save['records_over_time'].append({'game': agent.n_games, 'record_score': record})

                print('Game', agent.n_games, 'Score', score, 'Record:', record, f'Batch Loss: {batch_loss:.4f}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)

                plot_losses.append(batch_loss)
                total_loss += batch_loss
                mean_loss_val = total_loss / agent.n_games
                plot_mean_losses.append(mean_loss_val)

                # ===== 修改：將數據發送到佇列，而不是直接繪圖 =====
                plot_data = {
                    'scores': list(plot_scores),
                    'mean_scores': list(plot_mean_scores),
                    'losses': list(plot_losses),
                    'mean_losses': list(plot_mean_losses)
                }
                plot_queue.put(plot_data)
                # ===============================================

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving data and closing plot...")
    finally:
        # ===== 新增：通知繪圖進程結束並等待其關閉 =====
        plot_queue.put(None) # 發送停止信號
        plot_proc.join() # 等待繪圖進程完全結束
        # ==========================================

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.join(training_data_folder, f'training_data_dynamic_{timestamp}.json')
        
        data_to_save['scores'] = plot_scores
        data_to_save['mean_scores'] = plot_mean_scores
        data_to_save['losses'] = plot_losses
        data_to_save['mean_losses'] = plot_mean_losses
        data_to_save['total_games_played'] = agent.n_games
        data_to_save['final_record'] = record

        with open(file_name, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Training data saved to {file_name}")


if __name__ == '__main__':
    # Pygame 和 multiprocessing 在某些系統（如 macOS, Windows）上需要特別處理
    # 'spawn' 是更穩定但稍慢的啟動方式，建議使用
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    train()