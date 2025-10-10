import random
from pathlib import Path
from random import shuffle

import gradio as gr


# Load dictionary from utils/dic.txt
def load_dictionary():
    dict_path = Path(__file__).parent / "dict.txt"  # utils/dic.txt
    if dict_path.exists():
        words = {
            w.strip().upper()
            for w in dict_path.read_text(encoding="utf-8").splitlines()
            if w.strip().isalpha()
        }
        print(f"✅ Loaded {len(words):,} words from {dict_path}")
        return words
    else:
        print("⚠️ Dictionary file not found — using fallback wordlist.")
        return {"HELLO", "WORLD", "AI", "CODE", "TEXT", "PLAY", "PYTHON"}


DICTIONARY = load_dictionary()


LETTER_VALUES = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
    "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
    "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10, "#": 0
}


class Tile:
    def __init__(self, letter):
        self.letter = letter.upper()
        self.score = LETTER_VALUES.get(self.letter, 0)


class Bag:
    def __init__(self):
        self.tiles = []
        self.initialize_bag()

    def initialize_bag(self):
        freq = {
            "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2,
            "I": 9, "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2,
            "Q": 1, "R": 6, "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1,
            "Y": 2, "Z": 1, "#": 2
        }
        for la, n in freq.items():
            for _ in range(n):
                self.tiles.append(Tile(la))
        shuffle(self.tiles)

    def draw(self):
        return self.tiles.pop() if self.tiles else None


class Rack:
    def __init__(self, bag):
        self.bag = bag
        self.tiles = [bag.draw() for _ in range(7)]

    def refill(self):
        while len(self.tiles) < 7 and self.bag.tiles:
            self.tiles.append(self.bag.draw())

    def letters(self):
        return "".join(t.letter for t in self.tiles)


class Player:
    def __init__(self, name, bag, ai=False):
        self.name = name
        self.bag = bag
        self.rack = Rack(bag)
        self.score = 0
        self.ai = ai

    def play_word(self, word):
        word = word.upper()
        if all(word.count(ch) <= self.rack.letters().count(ch) for ch in word):
            for ch in word:
                for t in self.rack.tiles:
                    if t.letter == ch:
                        self.rack.tiles.remove(t)
                        break
            self.score += sum(LETTER_VALUES[c] for c in word)
            self.rack.refill()
            return True
        return False

    def ai_play(self):
        # AI finds first playable word from its rack that exists in dictionary
        rack_letters = self.rack.letters()
        possible = [w for w in DICTIONARY if all(w.count(c) <= rack_letters.count(c) for c in w)]
        if not possible:
            return None
        word = random.choice(possible)
        self.play_word(word)
        return word


class Game:
    def __init__(self):
        self.bag = Bag()
        self.board = []  # simplified for now
        self.player = Player("You", self.bag)
        self.computer = Player("Computer", self.bag, ai=True)
        self.turn = "You"

    def play_turn(self, word=None):
        if self.turn == "You":
            if not word:
                return "Enter a word!"
            word = word.upper()
            if word not in DICTIONARY:
                return f"'{word}' not found in dictionary."
            if not self.player.play_word(word):
                return "You don't have the tiles for that word!"
            result = f"You played {word} (+{sum(LETTER_VALUES[c] for c in word)} pts)"
            self.turn = "Computer"
        else:
            word = self.computer.ai_play()
            if not word:
                result = "Computer skips turn."
            else:
                result = f"Computer played {word} (+{sum(LETTER_VALUES[c] for c in word)} pts)"
            self.turn = "You"
        return result

    def scores(self):
        return f"You: {self.player.score} | Computer: {self.computer.score}"


# -------------------------------
# 🎮 Gradio UI for Scrabble Free Play
# -------------------------------


game = Game()


def play_turn(word):
    """Handles the player's turn followed immediately by AI turn"""
    # --- Player's turn ---
    if not word:
        return "Enter a word!", game.scores(), " ".join(game.player.rack.letters())
    word = word.upper()
    if word not in DICTIONARY:
        return f"'{word}' not found in dictionary.", game.scores(), " ".join(game.player.rack.letters())
    if not game.player.play_word(word):
        return "You don't have the tiles for that word!", game.scores(), " ".join(game.player.rack.letters())

    player_result = f"You played {word} (+{sum(LETTER_VALUES[c] for c in word)} pts)"

    # --- AI's turn ---
    ai_word = game.computer.ai_play()
    if ai_word:
        ai_result = f"Computer played {ai_word} (+{sum(LETTER_VALUES[c] for c in ai_word)} pts)"
    else:
        ai_result = "Computer skips turn."

    # Combine results
    result = player_result + "\n" + ai_result
    scores_text = game.scores()
    rack_text = " ".join(game.player.rack.letters())

    return result, scores_text, rack_text


def new_game():
    """Reset the game"""
    global game
    game = Game()
    return "New game started!", game.scores(), " ".join(game.player.rack.letters())


with gr.Blocks() as scrabble_demo:

    gr.Markdown("### Rules")
    gr.Markdown("You are competing with the computer to obtain as much points as possible.\n "
                "Score points by forming words from the characters in the rack, Highest score wins.🎉🎊")

    # --- Player Info Row ---
    with gr.Row():
        scores = gr.Textbox(label="Scores", value=game.scores(), interactive=False)
        rack_box = gr.Textbox(label="Your Rack", value=" ".join(game.player.rack.letters()), interactive=False)

    # --- Play controls ---
    with gr.Row():
        word_input = gr.Textbox(label="Enter your word", placeholder="Type a word using your rack letters...")
        submit_btn = gr.Button("Submit")
        new_btn = gr.Button("New Game", variant="secondary")

    # --- Output section ---
    result_box = gr.Textbox(label="Game Log", interactive=False)

    # --- Event bindings ---
    submit_btn.click(play_turn, inputs=word_input, outputs=[result_box, scores, rack_box])
    new_btn.click(new_game, outputs=[result_box, scores, rack_box])
