# prompts.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

# === Prompt Constants =========================================================

PLANNER_PROMPT = """
You are an expert content strategist who architects article outlines to meet precise specifications.

Your primary task is to create a well-scoped and balanced outline for an article on the topic below.
The structure of the outline itself must be designed to guide a writer to
produce a final article of approximately {word_count} words.

Topic:
{topic}

### Internal Thought Process (Follow this logic):

1. A {word_count}-word article needs a clear, focused structure. It cannot cover everything.
2. The Introduction and Conclusion will require about 10-15% of the total word count each.
3. The remaining word count must be distributed evenly across the main body sections.
4. To achieve this, I will create an outline with a limited number of points.
    An overly complex outline will make it impossible to meet the word count.

### Constraints for the Outline:

- The outline MUST include an **Introduction** and a **Conclusion**.
- Each main section MUST contain **no more than 2-3 subsections** or **key points**.
- The points in the outline should be concise phrases or topics, not full sentences or paragraphs.
    They are guideposts for the writer.

Task:
Generate the outline for the specified topic, strictly adhering to
the constraints above. Output only the clear, structured outline.
""".strip()


RESEARCHER_INSIGHTS_PROMPT = """
You are a research assistant. Provide comprehensive information for this query.

QUERY: {query}

Evidence:
{evidence}

Provide 3-4 key facts, insights, or perspectives. Include:
- Recent developments or trends
- Important statistics or data points
- Expert opinions or perspectives
- Relevant examples or case studies

Format each insight as a separate paragraph.
""".strip()


REWRITER_PROMPT = """
Revise the following article based on the feedback.

Original Article:
{article}

Feedback:
{feedback}

Outline:
{outline}

Instructions:
- Fix all issues noted in the feedback while preserving the article’s core content and structure.
- Top-level sections from the outline (e.g. Introduction, Main Body sections, Conclusion, References) must use #.
- If the outline includes subsections, use ## under their parent section.
- Do not use deeper levels (### or beyond).
- Ensure the article always includes # Introduction, # Conclusion, and # References as section headers.
- Keep style professional and engaging.
- Ensure final length is between {min_words} and {max_words} words.
- If over {max_words}, tighten language and remove redundancies until it reaches {max_words}.
- If under {min_words}, expand with more detail, examples, or smoother transitions until it reaches {max_words}.
- Ensure that it contains an **Introduction** and a **Conclusion** section with **References**

## IMPORTANT: FINAL ARTICLE MUST BE BETWEEN {min_words} AND {max_words}

Task:
Return only the revised article in markdown format only.
Output should not include code blocks e.g. "```markdown ```" or any additional text besides the article.
""".strip()


VALIDATOR_PROMPT = """
Review this article for quality.

ARTICLE:
{article}

Check:
1. Factual accuracy and consistency
2. Logical flow and coherence
3. Grammar and style
4. Contains an **Introduction** and **Conclusion**
4. Word count (target: {min_words}-{max_words} words, current: {word_count})

If acceptable, respond "VALID".
If issues exist, respond "REVISE: [specific issues]".

Be reasonably lenient - only reject for major problems.
""".strip()


WRITER_PROMPT = """
You are an expert content strategist and meticulous writer.
Your core skill is crafting high-quality articles that perfectly align with client specifications,
especially structural and length requirements.

Primary Objective:
Write a professional and engaging article on the provided topic.
The final output must be strictly between {min_words} and {max_words} words.
Adherence to this word count range is the most critical measure of success for this task.

Topic:
{topic}

Outline:
{outline}

Evidence:
{evidence}

Your Thought Process and Execution Plan (Follow these steps internally):
1. **Plan & Allocate:** First, analyze the outline. Mentally assign a target word count to each section
    (Introduction, each point in the outline, Conclusion) to ensure the total will fall
    within the {min_words}-{max_words} range. This allocation is your roadmap.
    For example, if the target is 1000 words and there are 5 sections, you might allocate roughly 200 words per section.
2. **Draft with Intent:** Write the article section by section, keeping your allocated word counts in mind.
    As you write, focus on integrating the research insights naturally. Use concrete examples to illustrate points.
    Ensure the writing is clear, professional, and engaging.
3. **Verify & Refine:** After completing the first draft, perform a word count.
4. If the draft is below {min_words}: Re-read the article and identify sections that can be expanded.
    Add more detail, provide deeper explanation, or introduce another supporting example
    to naturally increase the length without adding fluff.
5. If the draft is above {max_words}: Re-read the article and identify areas where you can be more concise.
    Eliminate redundant sentences, combine ideas, and remove less critical details to
    bring the word count down without sacrificing key information.

Repeat this verification and refinement step until the article's
word count is definitively within the {min_words}-{max_words} range.

Final Output Instructions:
1. The article must be between {min_words} and {max_words} words.
2. Top-level sections from the outline (e.g. Introduction, Main Body sections, Conclusion, References) must use #.
3. If the outline includes subsections, use ## under their parent section.
4. Do not use deeper levels (### or beyond).
5. Ensure the article always includes # Introduction, # Conclusion, and # References as section headers.
6. Use only simple markdown: headers, bold, italics, tables, blockquotes.
7. The article must contain an Introduction and a Conclusion.
8. Include a final references section containing the sources
9. The final output should be only the full article in markdown.
    Do not include your thought process, code blocks, notes, or any other explanations in the final response.

At the end, add a markdown References section listing the provided sources:

## References
{sources}

Write the complete article now, fully formatted in markdown, strictly following the outline and instructions.
Return only the markdown text with no code block wrappers, no tags, and no explanation.

## IMPORTANT: FINAL ARTICLE MUST BE BETWEEN {min_words} AND {max_words}

Task:
Write the full article now, following the plan above.
Keep the article concise and succinct, no need for super long sections.
Output should not include code blocks e.g. "```markdown ```" or any additional text besides the article.
""".strip()


# === Prompt Manager ===========================================================

class PromptManager:
    """Manages prompts from text files - NEVER overwrites existing files."""
    DEFAULT_PROMPTS = {
        "planner.txt": PLANNER_PROMPT,
        "researcher_insights.txt": RESEARCHER_INSIGHTS_PROMPT,
        "writer.txt": WRITER_PROMPT,
        "validator.txt": VALIDATOR_PROMPT,
        "rewriter.txt": REWRITER_PROMPT,
    }

    def __init__(self, prompts_path: str):
        self.prompts_path = Path(prompts_path)
        self.prompts_path.mkdir(parents=True, exist_ok=True)
        self._create_default_prompts()

    def _create_default_prompts(self):
        """Create default prompts ONLY if files don't exist."""
        created = 0
        for filename, content in self.DEFAULT_PROMPTS.items():
            filepath = self.prompts_path / filename
            if not filepath.exists():
                filepath.write_text(content.strip(), encoding="utf-8")
                created += 1
        if created > 0:
            print(f"✓ Created {created} default prompt files in {self.prompts_path}")
        else:
            print(f"✓ Using existing prompts from {self.prompts_path}")

    def get(self, prompt_name: str, **kwargs) -> str:
        """Load and format a prompt from file (expects '{name}.txt')."""
        filepath = self.prompts_path / f"{prompt_name}.txt"
        if not filepath.exists():
            raise FileNotFoundError(f"Prompt not found: {filepath}")
        template = filepath.read_text(encoding="utf-8")
        return template.format(**kwargs) if kwargs else template

    def save_version(self, prompt_name: str, suffix: str | None = None):
        """Save current prompt as versioned backup."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.prompts_path / "archive"
        archive_dir.mkdir(exist_ok=True)
        source = self.prompts_path / f"{prompt_name}.txt"
        dest = archive_dir / f"{prompt_name}_{suffix}.txt"
        dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"✓ Archived: {dest.name}")
