import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

load_dotenv(override=True)

API_KEY = os.getenv('GOOGLE_API_KEY', 'NOT FOUND')
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

print("✅ API Key loaded")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=API_KEY,
    temperature=0.1
)

# ---------------- Prompts ---------------- #
outline_prompt = PromptTemplate.from_template("""
Write an outline for an article on {topic}. The outline should be bullet points
starting with Introduction and ending with Conclusion. There should be 7 points in total.
""")

content_prompt = PromptTemplate.from_template("""
You are a specialist in writing scientific articles. Write an article
that is of professional standard based on the outline below.
Outline:
{outline}

Your article MUST be exactly around 1000 words (between 950-1050 words).
""")

decrease_editor_prompt = PromptTemplate.from_template("""
You are a professional text editor. Edit the article below by reducing the
number of words to **exactly between 950-1050 words** while preserving meaning and flow.

Current word count: {word_count}
Target: 950-1050 words

-----------------------------------------------------------------
{article}
-----------------------------------------------------------------

Return ONLY the edited article, no explanations.
""")

increase_editor_prompt = PromptTemplate.from_template("""
You are a professional text editor. Edit the article below by increasing the
number of words to **exactly between 950-1050 words** while maintaining clarity and coherence.

Current word count: {word_count}
Target: 950-1050 words

-----------------------------------------------------------------
{article}
-----------------------------------------------------------------

Return ONLY the edited article, no explanations.
""")

# ---------------- Chains ---------------- #
outline_chain = outline_prompt | llm | StrOutputParser()
content_chain = content_prompt | llm | StrOutputParser()
decrease_editor_chain = decrease_editor_prompt | llm | StrOutputParser()
increase_editor_chain = increase_editor_prompt | llm | StrOutputParser()

# ---------------- Pipeline ---------------- #
article_pipeline = {"outline": outline_chain} | RunnablePassthrough.assign(
    content=lambda x: content_chain.invoke({"outline": x["outline"]})
)


# ---------------- Word Counting Utility ---------------- #
def count_words(text):
    return len(re.findall(r"\b\w+\b", text))


# ---------------- Main Function with Loop ---------------- #
def get_article(topic, max_iterations=5):
    """
    Generate an article and ensure it's within 950-1050 words.
    
    Args:
        topic: The topic for the article
        max_iterations: Maximum number of editing attempts (default: 5)
    """
    # Step 1: Generate outline + article
    response = article_pipeline.invoke({"topic": topic})
    article = response["content"]

    # Step 2: Iteratively adjust length
    iteration = 0
    while iteration < max_iterations:
        word_count = count_words(article)
        print(f"📝 Iteration {iteration + 1}: Article length = {word_count} words")

        # Check if within acceptable range
        if 950 <= word_count <= 1050:
            print(f"✅ Article within acceptable range ({word_count} words)")
            return article, article

        # Adjust based on length
        if word_count > 1050:
            print(f"🔻 Article too long ({word_count} words) — reducing length...")
            article = decrease_editor_chain.invoke({
                "article": article,
                "word_count": word_count
            })
        else:  # word_count < 950
            print(f"🔺 Article too short ({word_count} words) — expanding length...")
            article = increase_editor_chain.invoke({
                "article": article,
                "word_count": word_count
            })

        iteration += 1
        if iteration >= 3:
            break

    # If max iterations reached, return best attempt
    final_word_count = count_words(article)
    print(f"⚠️ Max iterations reached. Final word count: {final_word_count}")
    return article


# ---------------- Run Test ---------------- #
if __name__ == "__main__":
    final_article, word_count = get_article("MLOps")
    print("\n\n" + "=" * 60)
    print(f"📄 Final Article ({word_count} words):\n")
    print("=" * 60)
    print(final_article)
