import gradio as gr
from textblob import TextBlob
import re

# ==================== TEXT CORRECTION FUNCTION ====================
def correct_text(text):
    """Corrects spelling and grammar using TextBlob"""
    if not text or text.strip() == "":
        return ""
    try:
        blob = TextBlob(text)
        corrected = str(blob.correct())
        return corrected
    except:
        return text

# ==================== CASE CONVERTER FUNCTIONS ====================
def convert_case(text, case_type):
    """Converts text to uppercase, lowercase, or title case"""
    if not text:
        return ""
    if case_type == "Uppercase":
        return text.upper()
    elif case_type == "Lowercase":
        return text.lower()
    elif case_type == "Title Case":
        return text.title()
    return text

# ==================== TEXT REVERSER FUNCTIONS ====================
def reverse_text(text, reverse_words, reverse_chars):
    """Reverses text based on selected options"""
    if not text:
        return ""
    
    result = text
    
    if reverse_words:
        # Reverse word order
        words = result.split()
        result = ' '.join(reversed(words))
    
    if reverse_chars:
        # Reverse all characters
        result = result[::-1]
    
    return result

# ==================== TEXT ANALYZER FUNCTIONS ====================
def analyze_text(text):
    """Analyzes text and returns statistics"""
    if not text or text.strip() == "":
        return "No text to analyze"
    
    # Character count (including spaces)
    char_count = len(text)
    
    # Word count
    words = text.split()
    word_count = len(words)
    
    # Average word length
    if word_count > 0:
        total_word_length = sum(len(word.strip('.,!?;:')) for word in words)
        avg_word_length = total_word_length / word_count
    else:
        avg_word_length = 0
    
    # Format output
    analysis = f"""
📊 **Text Analysis Results:**
━━━━━━━━━━━━━━━━━━━━━━━━━━
• **Word Count:** {word_count}
• **Character Count:** {char_count}
• **Average Word Length:** {avg_word_length:.2f} characters
━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    return analysis.strip()

# ==================== MAIN PROCESSING FUNCTION ====================
def process_text(original_text, corrected_text, case_type, reverse_words, reverse_chars):
    """Main function that processes text based on selected operations"""
    
    # Use corrected text for operations
    working_text = corrected_text if corrected_text else original_text
    
    if not working_text:
        return "", "", ""
    
    # Section 1: Case Converter
    case_result = convert_case(working_text, case_type)
    
    # Section 2: Text Reverser
    reverser_result = reverse_text(working_text, reverse_words, reverse_chars)
    
    # Section 3: Text Analyzer
    analyzer_result = analyze_text(working_text)
    
    return case_result, reverser_result, analyzer_result

# ==================== CLEAR FUNCTION ====================
def clear_all():
    """Clears all inputs and outputs"""
    return "", "", "", "", "", "Uppercase", False, False