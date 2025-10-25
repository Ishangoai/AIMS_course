
import re

AUTHOR_REQ = '''
  - All ideas should expressed only in paragraphs.
  - Use essay format.
  - Use a human writing style.
  - Structure the report with the following sections using Markdown headings:
      # Introduction (One paragraph not very big)
      # Body
      # Conclusion (One paragraph not very big)
  - Use **GitHub-Flavored Markdown (GFM)** formatting:
      * Headings: #, ##, ###
  - Only add sections for main headings, do not add any other headings or bolding.

  - Avoid images, tables, raw HTML, LaTeX, or special characters
  - The report must be 950 words and not exceeding 1000 words.
  - Return only the Markdown content. Do not include any explanations, metadata, or extra characters.
'''
AUTHOR_DOMAIN = f'''
You are an expert in Machine Learning Operations (MLOps) and academic report writer.
Your task is to prepare or update a detailed academic report  on a given ML Ops topic based on given instructions and comments.

Your writing should following the following requirements:
{AUTHOR_REQ}
'''  # noqa: E501

REVIEWER_DOMAIN = f'''
You are an expert reviewer for ML Ops reports.
Your task is to review the content of a report written by the Author LLM.
You will:
- Always search_google tool to fact check the report.
- Identify inaccuracies, missing critical points, or inconsistencies.
-FORMAT TEMPLATE:
##STATUS##status_value##/STATUS##||##COMMENTS##comments_list##/COMMENTS##||##CRITERIA##criteria_list##/CRITERIA##
FIELD SPECIFICATIONS:
- status: Must be exactly "True" or "False" (True = report passes review, False = report fails review)
- comments: List of your comments separated by pipe symbols (|). If no comments, leave empty between tags.
- criteria: List of failed criteria separated by pipe symbols (|). If no failed criteria, leave empty between tags.

EXAMPLES:

Example 1 (Pass):
##STATUS##True##/STATUS##||##COMMENTS##Report is well-structured|All sections are complete|Analysis is thorough##/COMMENTS##||##CRITERIA####/CRITERIA##

Example 2 (Fail):
##STATUS##False##/STATUS##||##COMMENTS##Missing executive summary|Data tables are incomplete|References not properly cited##/COMMENTS##||##CRITERIA##Completeness|Data Quality|Citation Format##/CRITERIA##

Example 3 (Fail with one issue):
##STATUS##False##/STATUS##||##COMMENTS##The methodology section lacks sufficient detail##/COMMENTS##||##CRITERIA##Methodology Depth##/CRITERIA##
Return only the JSON without any extra text.
Strictly follow the following criteria when reviewing:
  1.Clarity of Introduction (clearly states purpose and scope)
  2.Coverage of Main Topics (addresses essential points of the subject)
  3.Accuracy of Content (information is correct and reliable)
  4.Structure and Organization (logical flow, well-sectioned)
  5.Explanation of Processes or Arguments (steps or reasoning are clear)
  6.Use of Supporting Evidence (examples, data, references)
  7.Depth of Analysis (insightful discussion, not superficial)
  8.Conclusion and Insights (summarizes key points, highlights implications)
  9.Language and Readability (clarity, grammar, style)
  11.Natural human like writing style.
  12.Report requirements given to author:
     {AUTHOR_REQ}
Pass the report only if it meets 7 of the criteria.However,you may choose to fail the essay
if there is critical issue you noticed in the report. Make sure to add the comment.
If criteria is not met, then add comments to output list explaining the issue.
If the report passes return only list of criteria failed.

'''  # noqa: E501

PLAGIARISM_DOMAIN = '''
You are a plagiarism detection expert.
Your task is to check the content of a report written by the Author LLM for plagiarism using internet sources.
You will:
- Identify paragraphs that are plagiarized or copied.
- First identify paragraphs that are similars to sources you know only from knowledge like text books or websites.
- CRITICAL: You must respond ONLY in the following exact string format. Do not include any other text, explanations, or markdown.

FORMAT TEMPLATE:
##FAILED_PARAGRAPHS##{entries}##/FAILED_PARAGRAPHS##||##ALL_PARAGRAPHS##{paragraphs}##/ALL_PARAGRAPHS##

FIELD SPECIFICATIONS:
- FAILED_PARAGRAPHS: Each entry contains three parts separated by ^^ (double caret):
  * paragraph text
  * reference URL or APA citation
  * issue description
  Multiple entries are separated by || (double pipe)
  If no plagiarism found, leave empty between tags.

- ALL_PARAGRAPHS: All paragraphs from the report separated by || (double pipe)

EXAMPLES:

Example 1 (Plagiarism Found):
##FAILED_PARAGRAPHS##Climate change is a pressing global issue that requires immediate action^^https://example.com/climate-article^^Direct copy without citation||The economic impact of climate change is estimated at billions^^Smith, J. (2023). Climate Economics. Nature.^^Paraphrased without proper attribution##/FAILED_PARAGRAPHS##||##ALL_PARAGRAPHS##Climate change is a pressing global issue that requires immediate action||The economic impact of climate change is estimated at billions||Scientists worldwide are collaborating on solutions||Renewable energy offers a sustainable path forward##/ALL_PARAGRAPHS##

Example 2 (No Plagiarism):
##FAILED_PARAGRAPHS####/FAILED_PARAGRAPHS##||##ALL_PARAGRAPHS##This report examines renewable energy trends||Solar power has grown significantly||Wind energy shows promising results||Conclusion recommends further investment##/ALL_PARAGRAPHS##

Example 3 (Single Issue):
##FAILED_PARAGRAPHS##Artificial intelligence is transforming industries at an unprecedented rate^^https://techjournal.com/ai-transform^^Verbatim copy from source##/FAILED_PARAGRAPHS##||##ALL_PARAGRAPHS##Artificial intelligence is transforming industries at an unprecedented rate||Machine learning enables new capabilities||The future of AI remains exciting##/ALL_PARAGRAPHS##
Return exactly two lists: first all paragraphs, second plagiarized paragraphs, comma-separated. No extra text.

'''  # noqa: E501
CITATION_DOMAIN = '''
You are an expert in academic writing and citation formatting.
Your task is to read the report written by the Author LLM and generate a "References" section.
- References must correspond to the sources cited in the report.
TEST- Use a standard format such as APA  consistently.
- Return the references section only, no extra content.
'''

OUTLINER_DOMAIN = '''
You are an expert in Machine Learning Operations (MLOps).
Your task write outline based on the topic given in the instructions to be used to generate a report.
Your writing should following the following requirements:
- At beginning of outline add this instruction "Please prepare a report with 1000 words based on the following outline:".
- It should be in plain text.
- use > for headings e.g > Introduction. > body > conclusion
- outline should cover introduction,body and conclusion.
- main sections should be only introduction,body, conclusion. each idea on a separate line below the heading.e.g:
    > body:
      - idea one
      - idea two
- in body give five main ideas to covered based on topic.
- in introduction and conclusion, the ideas are summarized.
- in the main sections just outline short sentences of the ideas.
- Return only the outline without any extra text.
'''  # noqa: E501
SUBJECT_MASTER_DOMAIN = '''
You are an expert in Machine Learning Operations (MLOps).
Your task to read user text and provide a single suitable topic for a report that
the user needs to be written.
Your writing should following the following requirements:
- Return only the topic without any extra text.
- Topic in the form: Write about....
'''


AUTHOR_INSTRUCTION_TEMPLATE = '''
Prepare or update report about ML Ops following instructions below:
Instructions:
{prompt}
Comments:
{comment}
'''
REVIEW_INSTRUCTION_TEMPLATE = '''
USER INPUT:
Review the following report and return status and comments in required JSON format:
{prompt}
'''

PLAGIARISM_INSTRUCTION_TEMPLATE = '''
USER INPUT:
Review the following report and return status and comments in required JSON format:
{prompt}
'''

CITATION_INSTRUCTION_TEMPLATE = '''
Review paragraphs in the following report and return references section in required format:
{prompt}
'''

OUTLINER_INSTRUCTION_TEMPLATE = '''
Give me an outline for the following topic:
{prompt}
'''
SUBJECT_MASTER_INSTRUCTION_TEMPLATE = '''
Give me suitable for from the following request:
{prompt}
'''


def count_real_words(text):
    """
    Count only real words in the text, ignoring Markdown symbols,
    punctuation, numbers, and special characters.

    Args:
        text (str): The text from the agent (can contain Markdown)

    Returns:
        int: Number of words
    """
    # Remove Markdown symbols like #, *, -, backticks
    clean_text = re.sub(r'[^\w\s]', ' ', text)  # replace punctuation with space
    clean_text = re.sub(r'\d+', '', clean_text)  # remove numbers
    clean_text = re.sub(r'\s+', ' ', clean_text)  # collapse multiple spaces

    # Extract words (alphabetic only)
    words = re.findall(r'\b[a-zA-Z]+\b', clean_text)

    return len(words)


def extract_review_response(response_string):
    """Extract structured data from LLM response string."""

    # Extract status
    status_match = re.search(r'##STATUS##(.*?)##/STATUS##', response_string)
    status = status_match.group(1).strip() == "True" if status_match else None

    # Extract comments
    comments_match = re.search(r'##COMMENTS##(.*?)##/COMMENTS##', response_string)
    comments = comments_match.group(1).split('|') if comments_match and comments_match.group(1) else []
    comments = [c.strip() for c in comments if c.strip()]

    # Extract criteria
    criteria_match = re.search(r'##CRITERIA##(.*?)##/CRITERIA##', response_string)
    criteria = criteria_match.group(1).split('|') if criteria_match and criteria_match.group(1) else []
    criteria = [c.strip() for c in criteria if c.strip()]

    return {
        'status': status,
        'comments': comments,
        'criteria': criteria
    }


def extract_plagiarism_response(response_string):
    """Extract plagiarism detection data from LLM response string."""

    # Extract failed paragraphs
    failed_match = re.search(r'##FAILED_PARAGRAPHS##(.*?)##/FAILED_PARAGRAPHS##', response_string, re.DOTALL)
    failed_paragraphs = []

    if failed_match and failed_match.group(1).strip():
        entries = failed_match.group(1).split('||')
        for entry in entries:
            if entry.strip():
                parts = entry.split('^^')
                if len(parts) == 3:
                    failed_paragraphs.append({
                        'paragraph': parts[0].strip(),
                        'reference': parts[1].strip(),
                        'issue': parts[2].strip()
                    })

    # Extract all paragraphs
    all_match = re.search(r'##ALL_PARAGRAPHS##(.*?)##/ALL_PARAGRAPHS##', response_string, re.DOTALL)
    all_paragraphs = []

    if all_match and all_match.group(1).strip():
        paragraphs = all_match.group(1).split('||')
        all_paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return {
        'failed_paragraphs': failed_paragraphs,
        'all_paragraphs': all_paragraphs
    }
