import os
import random
import numpy as np
import time
import torch
from agent import LoadedNNAgent


class Game2048Env:
    def __init__(self):
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.action_space = ["w", "a", "s", "d"]
        self.reset()

    def reset(self):
        self.board = [[0] * 4 for _ in range(4)]
        self.add_new_tile()
        self.add_new_tile()
        return self._get_state()

    def _get_state(self):
        return [row[:] for row in self.board]

    @staticmethod
    def clear_screen():
        os.system("cls" if os.name == "nt" else "clear")

    def add_new_tile(self):
        empty = [(i, j) for i in range(4) for j in range(4) if self.board[i][j] == 0]
        if empty:
            i, j = random.choice(empty)
            self.board[i][j] = 2 if random.random() < 0.9 else 4

    def get_valid_moves(self):
        valid_moves = []
        original_board = [row[:] for row in self.board]

        for move in self.action_space:
            self.board = [row[:] for row in original_board]
            if self.move(move):
                valid_moves.append(move)

        self.board = original_board
        return valid_moves

    def print_board(self):
        self.clear_screen()
        print("\n2048\n")
        print(f"Total Score: {self.score}")
        print(f"Empty Tiles Bonus: {self.get_empty_tiles_score()}\n")
        for row in self.board:
            print("+------" * 4 + "+")
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("      |", end="")
                else:
                    print(f" {cell:4d} |", end="")
            print()
        print("+------" * 4 + "+")
        print("\nUse WASD to move (Q to quit)")

    def merge(self, row):
        new_row = [x for x in row if x != 0]
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                new_row.pop(i + 1)
            i += 1
        return new_row + [0] * (4 - len(new_row))

    def move(self, direction):
        moved = False
        board = [row[:] for row in self.board]

        if direction in "ws":
            board = list(map(list, zip(*board)))

        if direction in "sd":
            board = [row[::-1] for row in board]

        for i in range(4):
            row = self.merge(board[i])
            if row != board[i]:
                moved = True
                board[i] = row

        if direction in "sd":
            board = [row[::-1] for row in board]
        if direction in "ws":
            board = list(map(list, zip(*board)))

        self.board = board
        return moved

    def step(self, action):
        valid_moves = self.get_valid_moves()
        if action not in valid_moves:
            raise ValueError(f"Invalid move: {action}. Valid moves are: {valid_moves}")

        # Store old board for merge score calculation
        old_board = [row[:] for row in self.board]

        moved = self.move(action)
        self.add_new_tile()

        # Calculate composite score
        merge_score = self.get_merge_score(old_board, self.board)
        empty_score = self.get_empty_tiles_score()
        turn_score = merge_score + empty_score

        self.score += turn_score  # Update running total
        done = len(self.get_valid_moves()) == 0

        return self._get_state(), turn_score, done

    def get_merge_score(self, old_board, new_board):
        """Calculate score from merged tiles"""
        old_sum = sum(sum(row) for row in old_board)
        new_sum = sum(sum(row) for row in new_board)
        return new_sum - old_sum if new_sum > old_sum else 0

    def get_empty_tiles_score(self):
        """Reward for keeping spaces open"""
        empty_count = sum(1 for row in self.board for cell in row if cell == 0)
        return empty_count * 2


def play_game():
    env = Game2048Env()
    done = False
    score = 0

    print("\nControls: WASD to move, Q to quit")
    while not done:
        env.print_board()
        move = input("Enter move (WASD) or Q to quit: ").lower()

        if move == "q":
            break

        # Check if move is valid before attempting
        valid_moves = env.get_valid_moves()
        if move not in valid_moves:
            print("Invalid move! Valid moves are:", ", ".join(valid_moves))
            continue

        _, reward, done = env.step(move)
        score += reward

        if done:
            env.print_board()
            print(f"\nGame Over! Final score: {score}")
            print(f"Highest tile: {max(max(row) for row in env.board)}")
            break


def play_with_agent(delay=0.5):
    # Initialize environment and agent
    env = Game2048Env()
    agent = LoadedNNAgent()

    # Load trained weights
    try:
        agent.load_state_dict(torch.load("best_agent.pt"))
    except FileNotFoundError:
        print("Error: agent.pt file not found!")
        return

    # Set agent to evaluation mode
    agent.eval()

    # Play game
    state = env.reset()
    done = False
    total_score = 0
    moves = 0

    print("\nStarting game with trained agent...")
    time.sleep(1)

    while not done:
        # Clear screen and print current state
        env.print_board()
        print(f"\nMoves: {moves} | Score: {total_score}")

        # Get action from agent
        action = agent.get_action(state, env)
        if action is None:
            break

        # Execute action
        state, reward, done = env.step(action)
        total_score += reward
        moves += 1

        # Wait before next move
        time.sleep(delay)

    # Print final state
    env.print_board()
    print(f"\nGame Over!")
    print(f"Final Score: {total_score}")
    print(f"Total Moves: {moves}")
    print(f"Highest Tile: {max(max(row) for row in env.board)}")


if __name__ == "__main__":
    play_game()
    # play_with_agent(delay=0.2)
