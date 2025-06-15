# In snake_origin/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 設定 device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.to(device) # 模型實例化後再移至 device

    def forward(self, x):
        # x = x.to(device) # 輸入張量在傳入前處理
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        # 保存模型前移回 CPU 是個好習慣，以增加兼容性
        # self.to('cpu')
        torch.save(self.state_dict(), file_name)
        # self.to(device) # 移回原 device


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device) # 將模型移至 device
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # MSELoss 不需要顯式移至 device

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # done 是布林元組，保留在 CPU

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new # .item() 將結果移回 CPU
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # target 和 pred 都在 device 上
        loss.backward()
        self.optimizer.step()
        
        return loss.item() # 返回 loss 值