import multiprocessing as mp
import random
from game import Game2048Env
import matplotlib.pyplot as plt
import torch
from agent import NNAgent
import numpy as np
import time
from functools import wraps


class Logger:
    enabled = False

    @staticmethod
    def log(msg):
        if Logger.enabled:
            print(msg)


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not Logger.enabled:
            return func(*args, **kwargs)

        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        Logger.log(f"{func.__name__}: {duration:.2f}s")
        return result

    return wrapper


class AgentPopulation:
    def __init__(self, population_size=100):
        self.population_size = population_size
        self.mutation_rate = 0.3
        self.mutation_strength = 0.1
        self.action_space = ["w", "a", "s", "d"]
        self.agents = [NNAgent(self.action_space) for _ in range(population_size)]
        self.fitness_cache = {}  # Cache for fitness scores
        self.temperature = 1.0

    @timing_decorator
    def crossover(self, parent1: NNAgent, parent2: NNAgent) -> NNAgent:
        child = NNAgent(self.action_space)

        # Crossover neural network weights
        for child_param, p1_param, p2_param in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            # Random selection of weights from parents
            mask = torch.rand_like(child_param) < 0.5
            child_param.data = torch.where(mask, p1_param.data, p2_param.data)

        return child

    @timing_decorator
    def mutate_agent(self, agent: NNAgent) -> NNAgent:
        new_agent = NNAgent(self.action_space)
        new_agent.load_state_dict(agent.state_dict())

        mutation_strength = self.mutation_strength * (1.0 - self.temperature / 2.0)

        if random.random() < self.mutation_rate:
            for param in new_agent.parameters():
                noise = torch.randn_like(param) * mutation_strength
                mask = torch.rand_like(param) < 0.3  # Sparse mutations
                param.data += torch.where(mask, noise, torch.zeros_like(noise))

        return new_agent

    @timing_decorator
    def tournament_select(self, agents_with_fitness, k=3):
        """Select parent using tournament selection with cached fitness"""
        tournament = random.sample(agents_with_fitness, k)
        return max(tournament, key=lambda x: x[1][0])[
            0
        ]  # Return agent with highest score

    @timing_decorator
    def breed_new_generation(self, sorted_agents_with_fitness):
        new_agents = []
        elite_count = self.population_size // 5  # Keep top 20%

        # Elitism - keep best performers
        new_agents.extend(
            [agent for agent, _ in sorted_agents_with_fitness[:elite_count]]
        )

        # Breed remaining agents
        while len(new_agents) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_select(
                sorted_agents_with_fitness[: elite_count * 2]
            )
            parent2 = self.tournament_select(
                sorted_agents_with_fitness[: elite_count * 2]
            )

            # Crossover
            child = self.crossover(parent1, parent2)

            # Adaptive mutation - higher rate for worse performers
            position = len(new_agents) / self.population_size
            mut_rate = self.mutation_rate * (
                1 + position
            )  # Increase mutation for worse performers

            if random.random() < mut_rate:
                child = self.mutate_agent(child)

            new_agents.append(child)

        return new_agents


def evaluate_agent(agent: NNAgent, episodes=5) -> tuple[float, int]:
    env = Game2048Env()
    total_reward = 0
    max_tile = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.get_action(state, env)
            next_state, reward, done = env.step(action)
            total_reward += reward

            current_max = max(max(row) for row in env.board)
            max_tile = max(max_tile, current_max)

            state = next_state

    return total_reward / episodes, max_tile


def train_population(generations=500):
    population = AgentPopulation()
    chunk_size = population.population_size // mp.cpu_count()
    gen_times = []

    # Lists to store batch metrics
    batch_avg_rewards = []
    batch_max_tiles = []
    batch_highest_scores = []
    current_batch_scores = []
    current_batch_max = 0

    stagnation_counter = 0
    best_score = float("-inf")

    for generation in range(generations):
        gen_start = time.time()

        eval_start = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            fitness_results = pool.map(
                evaluate_agent, population.agents, chunksize=chunk_size
            )
        eval_time = time.time() - eval_start

        # Sort by fitness
        agents_with_fitness = list(zip(population.agents, fitness_results))
        agents_with_fitness.sort(key=lambda x: x[1][0], reverse=True)
        sorted_agents = [agent for agent, _ in agents_with_fitness]

        # Get elite scores for this generation
        elite_count = population.population_size // 40
        elite_scores = [score for _, (score, _) in agents_with_fitness[:elite_count]]
        current_batch_scores.extend(elite_scores)

        # Track highest score in batch
        gen_highest = max(score for _, (score, _) in agents_with_fitness)
        current_batch_max = max(current_batch_max, gen_highest)

        if (generation + 1) % 5 == 0:
            batch_avg_rewards.append(np.mean(current_batch_scores))
            batch_highest_scores.append(current_batch_max)
            batch_max_tiles.append(max(tile for _, (_, tile) in agents_with_fitness))

            # Reset batch tracking
            current_batch_scores = []
            current_batch_max = 0

            # Update plot
            plot_training_progress(
                batch_avg_rewards, batch_max_tiles, batch_highest_scores
            )

        print(
            f"Generation {generation}: Score = {gen_highest:.2f}, Max Tile = {max(tile for _, (_, tile) in agents_with_fitness)}"
        )

        current_best = max(score for _, (score, _) in agents_with_fitness)

        # Adjust temperature based on improvement
        if current_best > best_score:
            best_score = current_best
            stagnation_counter = 0
            population.temperature = max(0.5, population.temperature * 0.95)
        else:
            stagnation_counter += 1
            if stagnation_counter > 10:
                population.temperature = min(2.0, population.temperature * 1.1)

        breed_start = time.time()
        # Replace simple agent creation with proper breeding
        population.agents = population.breed_new_generation(agents_with_fitness)
        breed_time = time.time() - breed_start

        gen_time = time.time() - gen_start
        gen_times.append(gen_time)

        if Logger.enabled:
            Logger.log(f"\nGeneration {generation}:")
            Logger.log(f"Evaluation: {eval_time:.2f}s")
            Logger.log(f"Breeding: {breed_time:.2f}s")
            Logger.log(f"Total: {gen_time:.2f}s")
            Logger.log(f"Avg generation time: {sum(gen_times)/len(gen_times):.2f}s")

    return sorted_agents[0], (batch_avg_rewards, batch_max_tiles, batch_highest_scores)


def plot_training_progress(avg_rewards, max_tiles, highest_scores, save=False):
    if not plt.get_fignums():
        plt.ion()
        plt.figure(figsize=(12, 12))

    plt.clf()

    # Plot average elite rewards
    plt.subplot(3, 1, 1)
    plt.plot(range(10, len(avg_rewards) * 10 + 1, 10), avg_rewards, "b-")
    plt.title("Average Elite Rewards per 10 Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")

    # Plot highest scores
    plt.subplot(3, 1, 2)
    plt.plot(range(10, len(highest_scores) * 10 + 1, 10), highest_scores, "g-")
    plt.title("Highest Score per 10 Generations")
    plt.xlabel("Generation")
    plt.ylabel("Highest Score")

    # Plot max tiles
    plt.subplot(3, 1, 3)
    plt.plot(range(10, len(max_tiles) * 10 + 1, 10), max_tiles, "r-")
    plt.title("Maximum Tile Achieved per 10 Generations")
    plt.xlabel("Generation")
    plt.ylabel("Max Tile")

    plt.tight_layout()
    if save:
        plt.savefig("training_progress.png")
    else:
        plt.draw()
        plt.pause(0.001)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    best_agent, history = train_population(generations=1000)
    torch.save(best_agent.state_dict(), "best_agent.pt")
