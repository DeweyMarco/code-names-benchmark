import random
import csv
import os
from typing import List, Optional

from config import DataConfig, GameConfig

# Get the directory where this file is located
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CSV_PATH = os.path.join(_UTILS_DIR, "words.csv")

def load_words_from_csv(csv_path: Optional[str] = None) -> List[str]:
    """
    Load words from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing words (default: utils/words.csv)
    
    Returns:
        List of words from the CSV
    """
    if csv_path is None:
        csv_path = _DEFAULT_CSV_PATH
    
    words = []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            words.append(row['word'])
    return words


def generate_word_list(num_words: Optional[int] = None, csv_path: Optional[str] = None) -> List[str]:
    """
    Generate a list of random words.

    Args:
        num_words: Number of words in the list (default: from GameConfig.BOARD_SIZE)
        csv_path: Path to CSV file with word pool (default: utils/words.csv)

    Returns:
        List of random words
    """
    # Use default from config if not specified
    if num_words is None:
        num_words = GameConfig().BOARD_SIZE

    # Load word pool from CSV
    word_pool = load_words_from_csv(csv_path)

    # Ensure we have enough words
    data_config = DataConfig()
    if len(word_pool) < num_words:
        raise ValueError(f"Need at least {num_words} words in the pool")
    if len(word_pool) < data_config.MIN_WORDS_IN_POOL:
        print(f"Warning: Word pool has only {len(word_pool)} words, "
              f"recommended minimum is {data_config.MIN_WORDS_IN_POOL}")
    
    # Randomly sample words without replacement
    selected_words = random.sample(word_pool, num_words)
    
    return selected_words

def sort_words_to_codename_groups(words, config=None):
    """
    Sort the words in to codename groups

    Args:
        words: List of words to be sorted
        config: GameConfig instance (uses default if None)

    Returns:
        Dictionary with codename groups as keys and lists of words as values
        (all words normalized to lowercase for consistent comparison)
    """
    # Use default config if not provided
    if config is None:
        config = GameConfig()

    # Normalize words to lowercase for consistent comparison
    normalized_words = [w.lower() for w in words]

    # Assign words to groups based on config
    blue_end = config.BLUE_WORDS
    red_end = blue_end + config.RED_WORDS
    bomb_end = red_end + config.BOMB_COUNT

    blue_words = normalized_words[:blue_end]
    red_words = normalized_words[blue_end:red_end]
    bomb_word = normalized_words[red_end:bomb_end]
    neutral_words = normalized_words[bomb_end:]

    return {
        "blue": blue_words,
        "red": red_words,
        "bomb": bomb_word,
        "neutral": neutral_words
    }



def create_grid(selected_words, rows, cols):
    """
    Create the grid.
    
    Args:
        selected_words: List of words to be placed in the grid
        rows: Number of rows in the grid
        cols: Number of columns in the grid

    Returns:
        2D list containing the words in the grid
    """
    grid = []
    index = 0
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(selected_words[index])
            index += 1
        grid.append(row)
    
    return grid


def print_grid(grid):
    """
    Pretty print the word grid.
    
    Args:
        grid: 2D list of words
    """
    # Find the maximum word length for formatting
    max_length = max(len(word) for row in grid for word in row)
    
    for row in grid:
        print(" | ".join(word.ljust(max_length) for word in row))
        print("-" * (len(row) * (max_length + 3) - 3))

def print_codename_groups(codename_groups):
    """
    Print the codename groups.
    
    Args:
        codename_groups: Dictionary with codename groups as keys and lists of words as values
    """
    for group, words in codename_groups.items():
        print(f"{group}: {words}")

def print_color_grid(grid, codename_groups):
    """
    Print the color grid.

    Args:
        grid: 2D list of words
        codename_groups: Dictionary with codename groups as keys and lists of words as values
    """
    max_length = max(len(word) for row in grid for word in row)
    print("-" * (len(grid) * (max_length + 3) - 3))
    for row in grid:
        for word in row:
            word_lower = word.lower()
            if word_lower in codename_groups["blue"]:
                print(f"\033[94m{word.ljust(max_length)}\033[0m", end=" ")
            elif word_lower in codename_groups["red"]:
                print(f"\033[91m{word.ljust(max_length)}\033[0m", end=" ")
            elif word_lower in codename_groups["bomb"]:
                print(f"\033[93m{word.ljust(max_length)}\033[0m", end=" ")
            else:
                print(word.ljust(max_length), end=" ")
        print()


if __name__ == "__main__":
    # Generate and display a 5x5 grid of random words
    start_list = generate_word_list(25)
    
    # Assign words to codename groups
    codename_groups = sort_words_to_codename_groups(start_list)

    print_codename_groups(codename_groups)
    print("\n")
    print("=" * 60)
    random.shuffle(start_list)
    # Create grid with the same shuffled list
    grid = create_grid(start_list, 5, 5)
    print("5x5 Random Word Grid:")
    print_grid(grid)

    print_color_grid(grid, codename_groups)
    
