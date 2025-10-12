"""
Scrabble/Word Composing Game (Simplified)
--------------------------
A minimal interactive Word Composing game with Gradio UI.
You play against a computer to Score the most points through forming words from a given tray.

Authors: [Daniel Ayomide and Elisha Komolafe]
Date: 2025-10-12
"""

from pathlib import Path
from random import shuffle

import gradio as gr

# ============================================================
#  Utility Functions and Constants
# ============================================================


def load_dictionary() -> set[str]:
    """
    Load a list of valid Scrabble words from dict.txt.

    Returns:
        set[str]: A set of valid uppercase words.
    """
    dict_path = Path(__file__).parent / "dict.txt"
    if dict_path.exists():
        words = {
            w.strip().upper()
            for w in dict_path.read_text(encoding="utf-8").splitlines()
            if w.strip().isalpha()
        }
        print(f"✅ Loaded {len(words):,} words from {dict_path}")
        return words
    print("⚠️ Dictionary file not found — using fallback list.")
    return {"HELLO", "WORLD", "AI", "CODE", "TEXT", "PLAY", "PYTHON"}


# Load the dictionary once
DICTIONARY = load_dictionary()

# Scoring for each letter based on the scrabble game
LETTER_VALUES: dict[str, int] = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4,
    "I": 1, "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3,
    "Q": 10, "R": 1, "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8,
    "Y": 4, "Z": 10, "#": 0
}

# ============================================================
#  Game Components
# ============================================================


class Tile:
    """Represents a single letter tile with its corresponding score."""

    def __init__(self, letter: str):
        """
        Initialize a tile with a letter and its point value.

        Args:
            letter (str): Single character representing the tile.
        """
        self.letter: str = letter.upper()
        self.score: int = LETTER_VALUES.get(self.letter, 0)


class Bag:
    """Represents the bag of tiles that players draw from."""

    def __init__(self):
        """Initialize and shuffle the bag with standard Scrabble tile frequencies."""
        self.tiles: list[Tile] = []
        self._initialize_bag()

    def _initialize_bag(self) -> None:
        """Populate the bag with tiles according to standard Scrabble frequencies."""
        frequencies = {
            "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2,
            "I": 9, "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2,
            "Q": 1, "R": 6, "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1,
            "Y": 2, "Z": 1, "#": 2
        }
        for letter, count in frequencies.items():
            self.tiles.extend(Tile(letter) for _ in range(count))
        shuffle(self.tiles)

    def draw(self) -> Tile | None:
        """
        Draw a tile from the bag.

        Returns:
            Tile | None: A tile if available, else None.
        """
        return self.tiles.pop() if self.tiles else None


class Rack:
    """Represents a player's rack of up to 7 tiles."""

    def __init__(self, bag: Bag) -> None:
        """
        Initialize a rack by drawing up to 7 tiles from the bag.

        Args:
            bag (Bag): The bag to draw tiles from.
        """
        self.bag: Bag = bag
        self.tiles: list[Tile] = []
        self.refill()  # Fill the rack initially

    def refill(self) -> None:
        """Draw tiles until the rack has 7 letters (if possible)."""
        while len(self.tiles) < 7 and self.bag.tiles:
            tile = self.bag.draw()
            if tile is not None:
                self.tiles.append(tile)

    def letters(self) -> str:
        """Return a string of all letters currently on the rack."""
        return "".join(tile.letter for tile in self.tiles)


class Player:
    """Represents a player (human or AI) in the Scrabble game."""

    def __init__(self, name: str, bag: Bag, ai: bool = False):
        self.name: str = name
        self.bag: Bag = bag
        self.rack: Rack = Rack(bag)
        self.score: int = 0
        self.ai: bool = ai

    def play_word(self, word: str) -> bool:
        """
        Attempt to play a word using letters from the rack.

        Args:
            word (str): Word the player wants to play.

        Returns:
            bool: True if the word was successfully played.
        """
        word = word.upper()
        if all(word.count(ch) <= self.rack.letters().count(ch) for ch in word):
            for ch in word:
                for tile in self.rack.tiles:
                    if tile.letter == ch:
                        self.rack.tiles.remove(tile)
                        break
            self.score += sum(LETTER_VALUES[c] for c in word)
            self.rack.refill()
            return True
        return False

    def ai_play(self) -> str | None:
        """
        AI selects the **highest scoring valid word** from its rack.

        Returns:
            str | None: Word played, or None if no valid word found.
        """
        rack_letters = self.rack.letters()
        possible_words = [
            w for w in DICTIONARY
            if all(w.count(c) <= rack_letters.count(c) for c in w)
        ]
        if not possible_words:
            return None
        # Choose the highest scoring word
        best_word = max(possible_words, key=lambda w: sum(LETTER_VALUES[c] for c in w))
        self.play_word(best_word)
        return best_word


class Game:
    """Controller for the Scrabble game."""

    def __init__(self, max_rounds: int = 5):
        self.bag = Bag()
        self.player = Player("You", self.bag)
        self.computer = Player("Computer", self.bag, ai=True)
        self.round = 1
        self.max_rounds = max_rounds
        self.game_over = False

    def _execute_turn(self, player: Player, word: str | None = None) -> str:
        """
        Internal helper to execute a player's move.

        Args:
            player (Player): Player object (human or AI).
            word (str | None): Word to play (for human). Ignored for AI.

        Returns:
            str: Message describing the turn result.
        """
        if player.ai:
            ai_word = player.ai_play()
            return (
                "Computer skips turn."
                if not ai_word
                else f"Computer played {ai_word} (+{sum(LETTER_VALUES[c] for c in ai_word)} pts)"
            )

        if not word:
            return "Enter a word!"
        word = word.upper()
        if word not in DICTIONARY:
            return f"'{word}' not found in dictionary."
        if not player.play_word(word):
            return "You don't have the tiles for that word!"
        return f"You played {word} (+{sum(LETTER_VALUES[c] for c in word)} pts)"

    def play_turn(self, word: str | None = None) -> str:
        """
        Execute a full round: player and AI move.

        Args:
            word (str | None): Word entered by human.

        Returns:
            str: Combined log of both moves and game status.
        """
        if self.game_over:
            return "🎮 Game over! Please start a new game."

        player_result = self._execute_turn(self.player, word)
        ai_result = self._execute_turn(self.computer)

        self.round += 1
        if self.round > self.max_rounds:
            self.game_over = True
            player_result += "\n\n🏁 Game Over!"
            if self.player.score > self.computer.score:
                ai_result += "\n🎉 Congratulations, you win! 🏆"
            elif self.player.score < self.computer.score:
                ai_result += "\n🤖 Computer wins! Try again! 💪"
            else:
                ai_result += "\n🤝 It's a tie!"

        return f"{player_result}\n{ai_result}"

    def skip_turn(self) -> str:
        """Skip human turn, refill their rack, and let AI play."""
        if self.game_over:
            return "🎮 Game over! Please start a new game."

        # Clear the rack safely without reassigning
        self.player.rack.tiles.clear()
        self.player.rack.refill()

        player_result = "You skipped your turn."
        ai_result = self._execute_turn(self.computer)
        self.round += 1

        if self.round > self.max_rounds:
            self.game_over = True
            player_result += "\n\n🏁 Game Over!"
            if self.player.score > self.computer.score:
                ai_result += "\n🎉 Congratulations, you win! 🏆"
            elif self.player.score < self.computer.score:
                ai_result += "\n🤖 Computer wins! Try again! 💪"
            else:
                ai_result += "\n🤝 It's a tie!"

        return f"{player_result}\n{ai_result}"

    def scores(self) -> str:
        """
        Get a formatted score string.
        Returns:
            str: Scores of both players and current round.
        """
        return (
            f"You: {self.player.score} | Computer: {self.computer.score} | "
            f"Round: {min(self.round, self.max_rounds)}/{self.max_rounds}"
        )

# ============================================================
# 🎮 Gradio Interface Wrappers
# ============================================================


game = Game()


def gr_play_turn(word: str) -> tuple[str, str, str, str]:
    """
    Gradio wrapper to execute a turn.

    Args:
        word (str): Player's entered word.

    Returns:
        tuple[str, str, str, str]: Game log, scores, updated rack, cleared input.
    """
    result = game.play_turn(word)
    return result, game.scores(), " ".join(game.player.rack.letters()), ""


def new_game() -> tuple[str, str, str, str]:
    """
    Reset the Scrabble game and start a new session.

    Returns:
        tuple[str, str, str, str]: Reset message, scores, new rack, cleared input.
    """
    global game
    game = Game()
    return "New game started!", game.scores(), " ".join(game.player.rack.letters()), ""

# ============================================================
# 🧩 Gradio Interface
# ============================================================


with gr.Blocks() as scrabble_demo:

    gr.Markdown("## Game Rules")

    gr.Markdown(
        """
        **Objective:** Form valid dictionary words to score points.
        The player with the highest score after 5 rounds wins.

        **Gameplay:**
        1. Each player gets **5 turns**.
        2. Form **valid dictionary words** using the letters in your rack.
        3. Each letter has a point value; the word's score is the sum of its letters.
        4. Your Game log will appear as you play.

        **Buttons:**
        - **Submit:** Play the word you entered. If valid, your score updates and your rack refills with new tiles.
            If invalid, your rack stays the same.
        - **Skip Turn:** Skip your turn. Your rack refreshes with a new set of tiles, and the computer plays its turn.
        - **New Game:** Start a fresh game with empty scores and a new rack of tiles.

        **End of Game:** The game ends after **5 rounds** per player. The player with the highest score wins!
        """
    )

    with gr.Row():
        scores_box = gr.Textbox(label="Scores", value=game.scores(), interactive=False)
        rack_box = gr.Textbox(label="Your Rack", value=" ".join(game.player.rack.letters()), interactive=False)

    with gr.Row():
        word_input = gr.Textbox(label="Enter your word", placeholder="Type a valid word...")
        submit_btn = gr.Button("Submit")
        skip_btn = gr.Button("Skip Turn", variant="secondary")
        new_btn = gr.Button("New Game", variant="secondary")

    result_box = gr.Textbox(label="Game Log", interactive=False)

    # Attach events
    submit_btn.click(
        gr_play_turn,
        inputs=word_input,
        outputs=[result_box, scores_box, rack_box, word_input]
    )

    skip_btn.click(
        lambda: (game.skip_turn(), game.scores(), " ".join(game.player.rack.letters()), ""),
        outputs=[result_box, scores_box, rack_box, word_input]
    )

    new_btn.click(
        new_game,
        outputs=[result_box, scores_box, rack_box, word_input]
    )
