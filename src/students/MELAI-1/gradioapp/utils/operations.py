"""Operations for Text Processing App.

Location: MELAI/gradio_app/operations.py
Contains functional utilities used by app.py
"""

import datetime
import os
import re
import tempfile
from collections import Counter
from typing import Any, Dict, Tuple

# from fpdf import FPDF


class SpellChecker:
    """Simple spell checker using common misspellings dictionary."""

    def __init__(self) -> None:
        """Initialize spell checker with corrections dictionary."""
        self.corrections: Dict[str, str] = {
            "teh": "the",
            "hw": "how",
            "r": "are",
            "u": "you",
            "ur": "your",
            "wat": "what",
            "waht": "what",
            "wht": "what",
            "thier": "their",
            "recieve": "receive",
            "occured": "occurred",
            "seperate": "separate",
            "definately": "definitely",
            "wierd": "weird",
            "freind": "friend",
            "untill": "until",
            "begining": "beginning",
            "goverment": "government",
            "wich": "which",
            "realy": "really",
            "alot": "a lot",
            "dont": "don't",
            "doesnt": "doesn't",
            "cant": "can't",
            "wont": "won't",
            "shouldnt": "shouldn't",
            "wouldnt": "wouldn't",
            "couldnt": "couldn't",
            "isnt": "isn't",
            "arent": "aren't",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hadnt": "hadn't",
            "wasnt": "wasn't",
            "werent": "weren't",
            "thats": "that's",
            "whats": "what's",
            "hes": "he's",
            "shes": "she's",
            "its": "it's",
            "theyre": "they're",
            "youre": "you're",
            "were": "we're",
            "ive": "I've",
            "youve": "you've",
            "theyve": "they've",
            "weve": "we've",
            "im": "I'm",
            "id": "I'd",
            "youd": "you'd",
            "hed": "he'd",
            "shed": "she'd",
            "theyd": "they'd",
            "wed": "we'd",
            "ill": "I'll",
            "youll": "you'll",
            "hell": "he'll",
            "shell": "she'll",
            "theyll": "they'll",
            "well": "we'll",
            "payed": "paid",
            "mispell": "misspell",
            "greatful": "grateful",
            "unfortunatly": "unfortunately",
            "neccessary": "necessary",
            "occassion": "occasion",
            "accross": "across",
            "adress": "address",
            "arguement": "argument",
            "basicly": "basically",
            "beleive": "believe",
            "buisness": "business",
            "calender": "calendar",
            "catagory": "category",
            "cemetary": "cemetery",
            "changable": "changeable",
            "collegue": "colleague",
            "comming": "coming",
            "committment": "commitment",
            "concious": "conscious",
            "coperate": "cooperate",
            "critisism": "criticism",
            "developement": "development",
            "dissapear": "disappear",
            "embarass": "embarrass",
            "enviroment": "environment",
            "existance": "existence",
            "fourty": "forty",
            "harrass": "harass",
            "humourous": "humorous",
            "independant": "independent",
            "jewellery": "jewelry",
            "liason": "liaison",
            "maintainance": "maintenance",
            "occurance": "occurrence",
            "persue": "pursue",
            "possesion": "possession",
            "prefered": "preferred",
            "priviledge": "privilege",
            "profesional": "professional",
            "publically": "publicly",
            "reccomend": "recommend",
            "refered": "referred",
            "relevent": "relevant",
            "rythm": "rhythm",
            "shedule": "schedule",
            "succesful": "successful",
            "supress": "suppress",
            "temperture": "temperature",
            "tommorrow": "tomorrow",
            "truely": "truly",
            "useable": "usable",
            "vaccuum": "vacuum",
            "visable": "visible",
            "wensday": "wednesday",
            "whereever": "wherever",
            "lol": "laugh out loud",
            "brb": "be right back",
            "omg": "oh my god",
            "btw": "by the way",
            "idk": "I don't know",
            "tbh": "to be honest",
        }

    def _extract_punctuation(self, stripped: str) -> Tuple[str, str, int, int]:
        """Extract prefix and suffix punctuation from word.

        Returns:
            Tuple of (prefix, suffix, start_index, end_index)
        """
        prefix = ""
        start_idx = 0
        for i, char in enumerate(stripped):
            if char.isalpha():
                start_idx = i
                break
            prefix += char

        suffix = ""
        end_idx = len(stripped)
        for i in range(len(stripped) - 1, -1, -1):
            if stripped[i].isalpha():
                end_idx = i + 1
                break
            suffix = stripped[i] + suffix

        return prefix, suffix, start_idx, end_idx

    def _preserve_capitalization(self, core_word: str, corrected: str) -> str:
        """Preserve original word's capitalization in correction.

        Args:
            core_word: Original word
            corrected: Corrected word

        Returns:
            Corrected word with preserved capitalization
        """
        if not core_word or not core_word[0].isupper():
            return corrected

        if len(core_word) > 1 and core_word.isupper():
            return corrected.upper()

        return corrected.capitalize()

    def fix_word(self, word: str) -> str:
        """Fix a single word if it is in the corrections dictionary.

        Args:
            word: The word to check and potentially correct.

        Returns:
            The corrected word or the original word if no correction found.
        """
        if not word:
            return word

        stripped = word.strip()
        if not stripped:
            return word

        prefix, suffix, start_idx, end_idx = self._extract_punctuation(stripped)
        core_word = stripped[start_idx:end_idx]

        if not core_word:
            return word

        lower_word = core_word.lower()
        if lower_word not in self.corrections:
            return word

        corrected = self.corrections[lower_word]
        corrected = self._preserve_capitalization(core_word, corrected)

        return prefix + corrected + suffix


class TextCorrector:
    """Handle text correction operations."""

    def __init__(self) -> None:
        """Initialize text corrector with spell checker."""
        self.spell_checker = SpellChecker()

    def correct(self, text: str) -> str:
        """Perform comprehensive text correction.

        Args:
            text: The text to correct.

        Returns:
            The corrected text.
        """
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        corrected_words = [self.spell_checker.fix_word(word) for word in words]

        corrected = " ".join(corrected_words)

        corrected = re.sub(r"\s+([.,!?;:])", r"\1", corrected)
        corrected = re.sub(r"([.,!?;:])([A-Za-z])", r"\1 \2", corrected)

        sentences = re.split(r"([.!?]+)", corrected)
        result = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
                result.append(sentence)

                if i + 1 < len(sentences):
                    result.append(sentences[i + 1])
                elif sentence[-1] not in ".!?":
                    result.append(".")

        final = "".join(result)
        final = re.sub(r"([.!?])([A-Z])", r"\1 \2", final)

        return final if final else corrected


class TextConverter:
    """Handle text conversion operations."""

    @staticmethod
    def convert(text: str, mode: str) -> str:
        """Convert text based on mode.

        Args:
            text: The text to convert.
            mode: The conversion mode (Uppercase, Lowercase, Title Case).

        Returns:
            The converted text.
        """
        if not text:
            return ""
        if mode == "Uppercase":
            return text.upper()
        if mode == "Lowercase":
            return text.lower()
        if mode == "Title Case":
            return text.title()
        return text


class TextReverser:
    """Handle text reversal operations."""

    @staticmethod
    def reverse(text: str, mode: str = "words") -> str:
        """Reverse text based on mode.

        Args:
            text: The text to reverse.
            mode: The reversal mode (words or characters).

        Returns:
            The reversed text.
        """
        if not text:
            return ""

        if mode == "characters":
            return text[::-1]
        words = text.split()
        return " ".join(reversed(words))


class TextAnalyzer:
    """Handle text analysis operations."""

    @staticmethod
    def analyze(text: str) -> Dict[str, Any]:
        """Return comprehensive text analysis.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary containing analysis results.
        """
        if not text:
            return {"word_count": 0, "char_count": 0, "most_common": "N/A", "sentence_count": 0, "avg_word_length": 0}

        words = re.findall(r"\b\w+\b", text.lower())
        word_count = len(words)
        char_count = len(text)
        sentences = re.split(r"[.!?]+", text)
        sentence_count = len([s for s in sentences if s.strip()])

        if not words:
            return {
                "word_count": 0,
                "char_count": char_count,
                "most_common": "N/A",
                "sentence_count": sentence_count,
                "avg_word_length": 0,
            }

        freq = Counter(words)
        most_common = freq.most_common(1)[0][0] if freq else "N/A"
        avg_word_length = sum(len(w) for w in words) / len(words)

        return {
            "word_count": word_count,
            "char_count": char_count,
            "most_common": most_common,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 1),
        }


class WordCloudGenerator:
    """Generate visual word cloud."""

    @staticmethod
    def generate(text: str) -> str:
        """Generate HTML word cloud with varying sizes.

        Args:
            text: The text to generate word cloud from.

        Returns:
            HTML string representing the word cloud.
        """
        if not text:
            return "<p>No text to generate word cloud.</p>"

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return "<p>No words found.</p>"

        freq = Counter(words)
        top_words = freq.most_common(50)

        if not top_words:
            return "<p>No words to display.</p>"

        max_freq = top_words[0][1]
        min_freq = top_words[-1][1]

        html = '<div style="text-align: center; line-height: 2.5; '
        html += "padding: 20px; background: linear-gradient(135deg, "
        html += '#667eea 0%, #764ba2 100%); border-radius: 10px;">'

        colors = [
            "#FFD700",
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#FFA07A",
            "#98D8C8",
            "#F7DC6F",
            "#BB8FCE",
            "#85C1E2",
            "#F8B739",
        ]

        for idx, (word, count) in enumerate(top_words):
            if max_freq == min_freq:
                font_size = 24
            else:
                ratio = (count - min_freq) / (max_freq - min_freq)
                font_size = 12 + ratio * 36

            color = colors[idx % len(colors)]
            style = (
                f"font-size: {font_size}px; color: {color}; "
                f"margin: 5px; display: inline-block; "
                f"font-weight: bold; "
                f"text-shadow: 2px 2px 4px rgba(0,0,0,0.3);"
            )
            html += f'<span style="{style}">{word}</span> '

        html += "</div>"
        return html


class FileExporter:
    """Handle file export operations."""

    @staticmethod
    def _get_filename(ext: str) -> str:
        """Return an intuitive filename.

        Args:
            ext: The file extension.

        Returns:
            Full path to the temporary file.
        """
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"TextStudio_Export_{now}.{ext}"
        tmpdir = tempfile.gettempdir()
        return os.path.join(tmpdir, fname)

    @staticmethod
    def save_as_txt(content: str) -> str:
        """Save content as TXT file.

        Args:
            content: The content to save.

        Returns:
            Path to the saved file.
        """
        path = FileExporter._get_filename("txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # @staticmethod
    # def save_as_pdf(content: str) -> str:
    #     """Save content as PDF file.

    #     Args:
    #         content: The content to save.

    #     Returns:
    #         Path to the saved file.
    #     """
    #     path = FileExporter._get_filename("pdf")

    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_auto_page_break(auto=True, margin=15)
    #     pdf.set_font("Arial", size=10)

    #     lines = content.split("\n")
    #     for line in lines:
    #         try:
    #             line_encoded = line.encode("latin-1", "replace").decode("latin-1")
    #             pdf.multi_cell(0, 6, text=line_encoded)
    #         except Exception:
    #             pdf.multi_cell(0, 6, text=line.encode("ascii", "ignore").decode("ascii"))

    #     pdf.output(path)
    #     return path


# Public API functions
def correct_text(text: str) -> str:
    """Correct text using TextCorrector."""
    corrector = TextCorrector()
    return corrector.correct(text)


def convert_text(text: str, mode: str) -> str:
    """Convert text using TextConverter."""
    return TextConverter.convert(text, mode)


def reverse_text(text: str, mode: str = "words") -> str:
    """Reverse text using TextReverser."""
    return TextReverser.reverse(text, mode)


def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text using TextAnalyzer."""
    return TextAnalyzer.analyze(text)


def generate_word_cloud(text: str) -> str:
    """Generate word cloud using WordCloudGenerator."""
    return WordCloudGenerator.generate(text)


def save_as_txt(content: str) -> str:
    """Save as TXT using FileExporter."""
    return FileExporter.save_as_txt(content)


# def save_as_pdf(content: str) -> str:
#     """Save as PDF using FileExporter."""
#     return FileExporter.save_as_pdf(content)
