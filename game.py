import os
import random
import numpy as np


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
        for row in self.board:
            print("+------" * 4 + "+")
            print("|", end="")
            for cell in row:
                if cell == 0:
                    print("      |", end="")
                else:
                    print(f"{cell:^6}|", end="")
            print()
        print("+------" * 4 + "+")
        print("\nUse WASD to move (Q to quit)")

    def merge(self, row):
        # Remove zeros and merge similar numbers
        new_row = [x for x in row if x != 0]
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                new_row.pop(i + 1)
            i += 1
        # Pad with zeros
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

        moved = self.move(action)
        self.add_new_tile()

        # Use valid moves check directly
        done = len(self.get_valid_moves()) == 0

        if done:
            score = sum(sum(row) for row in self.board)
        else:
            score = 0

        return self._get_state(), score, done


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


if __name__ == "__main__":
    play_game()
