

import html
import logging
import os
import re

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .utils import (
    AUTHOR_DOMAIN,
    AUTHOR_INSTRUCTION_TEMPLATE,
    CITATION_DOMAIN,
    CITATION_INSTRUCTION_TEMPLATE,
    OUTLINER_DOMAIN,
    OUTLINER_INSTRUCTION_TEMPLATE,
    PLAGIARISM_DOMAIN,
    PLAGIARISM_INSTRUCTION_TEMPLATE,
    REVIEW_INSTRUCTION_TEMPLATE,
    REVIEWER_DOMAIN,
    SUBJECT_MASTER_DOMAIN,
    SUBJECT_MASTER_INSTRUCTION_TEMPLATE,
    count_real_words,
    extract_plagiarism_response,
    extract_review_response,
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)


@tool
def search_google(query: str) -> str:
    """
    Performs a Google search and returns the results.
    Use this to answer questions that require up-to-date or external information.
    """
    print("tool running ", query)
    """Search Google for latest information on topic."""
    results = search.results(query, 3) # type: ignore
    return results # type: ignore

class MyAgent():
    def __init__(self, domain, template, reviewer=False, parser=StrOutputParser(), temperature=0.0):
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature

        )
        self.str_parser = parser
        self.reviewer = reviewer

        if self.reviewer:
            self.prompt_template = PromptTemplate.from_template(template)
            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

            # Initialize agent with the tool
            agent = create_react_agent(
                self.llm,
                tools=[search_google],
                prompt=domain
            )
            self.agent = agent
        else:
            self.prompt_template = PromptTemplate.from_template(domain + template)
            self.chain = self.prompt_template | self.llm | self.str_parser

    def create_prompt(self, user_prompt, comment='None'):
        return {
            'prompt': user_prompt,
            'comment': comment
        }

    def create_review_prompt(self, user_prompt):
        review_prompt_text = self.prompt_template.format(prompt=user_prompt)

        return review_prompt_text

    def first_invoke(self, user_prompt, comment='None'):
        response = self.invoke(user_prompt, comment='None')
        return response

    def invoke(self, user_prompt, comment='None'):
        if self.reviewer:

            response = self.agent.invoke({"messages": self.create_review_prompt(user_prompt)})
            response = response["messages"][-1].content
        else:
            message = self.create_prompt(user_prompt, comment)
            response = self.chain.invoke(message)
        return response


# function generate llm review list
def generate_review(data):
    formatted_comments = "\n".join(data['comments'])
    formatted_criteria = "\n".join(data['criteria'])
    return '   key comments:' + formatted_comments + '   key criteria:' + formatted_criteria


# subject master /topic generator
subject_master = MyAgent(SUBJECT_MASTER_DOMAIN, SUBJECT_MASTER_INSTRUCTION_TEMPLATE, temperature=0.0)
# outliner
outliner = MyAgent(OUTLINER_DOMAIN, OUTLINER_INSTRUCTION_TEMPLATE, temperature=0.3)
# author
author = MyAgent(AUTHOR_DOMAIN, AUTHOR_INSTRUCTION_TEMPLATE, temperature=0.8)

# reviewer
reviewer = MyAgent(REVIEWER_DOMAIN, REVIEW_INSTRUCTION_TEMPLATE, reviewer=True, temperature=0)

# plagrism
plagrism_checker = MyAgent(PLAGIARISM_DOMAIN, PLAGIARISM_INSTRUCTION_TEMPLATE, reviewer=True, temperature=0.0)


def convert_report(md_text):
    # Escape HTML special chars
    text = html.escape(md_text)

    # Headings
    text = re.sub(r'###### (.*)', r'<h6>\1</h6>', text)
    text = re.sub(r'##### (.*)', r'<h5>\1</h5>', text)
    text = re.sub(r'#### (.*)', r'<h4>\1</h4>', text)
    text = re.sub(r'### (.*)', r'<h3>\1</h3>', text)
    text = re.sub(r'## (.*)', r'<h2>\1</h2>', text)
    text = re.sub(r'# (.*)', r'<h1>\1</h1>', text)

    # Bold **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # Italic *text*
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)

    # Links [text](url)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    # Newlines -> <br>
    text = text.replace('\n', '<br>\n')

    return text


# citation llm
citation_maker = MyAgent(CITATION_DOMAIN, CITATION_INSTRUCTION_TEMPLATE, reviewer=True, temperature=0.5)


def drafting_process(user_input, initialize=True, user_comment='None'):
    if (initialize):
        print("Analyzing topic...")

        subject_response = subject_master.invoke(user_input)

        print("Making Outling..")
        outline_response = outliner.invoke(subject_response)

        print("Drafting report..")
        content_response = author.invoke(outline_response)
    else:
        print("Drafting report..")
        content_response = author.invoke(user_input, user_comment)

    # author and reviewer in loop
    current_model = 'reviewer'
    models = {
        'author': author,
        'reviewer': reviewer
    }
    quota = 2

    response_loop = content_response
    comment = 'None'
    index = 0
    while (index < quota):
        if (current_model == 'author'):
            print("Redrafting report..")
            response = models[current_model].invoke(response_loop, comment)
            response_loop = response
            content_response = response
            current_model = 'reviewer'
            print("Author done", index)
        elif (current_model == 'reviewer'):
            print("Reviewing report..")
            response = models[current_model].invoke(response_loop)
            extract = extract_review_response(response)
            status = extract['status']
            print("Reviewer done", index)
            if status:
                print("status", status, extract)

                word_count = count_real_words(content_response)
                return content_response, str(word_count)

            else:
                current_model = 'author'
                response_loop = content_response
                comment = generate_review(extract)

        index += 1
    word_count = count_real_words(content_response)  # ignore
    return content_response, str(word_count)
    # return to user the content


# function to calculate llm plagrism score
def calculate_plagrism_score(extract):
    failed = extract['failed_paragraphs']
    all = extract['all_paragraphs']
    score = len(failed) / len(all)
    return score


def plagrism_check(user_input):
    # plagrism check
    plagrism_response = plagrism_checker.invoke(user_input)
    extract = extract_plagiarism_response(plagrism_response)
    score = calculate_plagrism_score(extract)
    return score


def citation_generator(user_input):

    citation_response = citation_maker.invoke(user_input)
    return citation_response
