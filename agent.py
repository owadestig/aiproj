import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NNAgent(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.action_map = {0: "w", 1: "a", 2: "s", 3: "d"}

        self.fc1 = nn.Linear(16, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 4)

        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x1 = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x2 = self.dropout(F.relu(self.ln2(self.fc2(x1))))
        x = x1 + x2

        x = self.dropout(F.relu(self.ln3(self.fc3(x))))
        x = F.relu(self.fc4(x))
        return self.fc5(x)

    def get_action(self, state, env):
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None  # Game is over if there are no valid moves

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.int32)
        state_tensor = torch.from_numpy(state.flatten()).float()
        state_tensor = state_tensor / 2048.0

        with torch.no_grad():
            q_values = self(state_tensor).squeeze(0)
            valid_indices = [
                i for i, move in self.action_map.items() if move in valid_moves
            ]

            valid_q = q_values[valid_indices]
            action_idx = valid_indices[valid_q.argmax().item()]
            return self.action_map[action_idx]
