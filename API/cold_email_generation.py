from langchain_groq import ChatGroq
import yaml
import os
from langchain_core.prompts import PromptTemplate
from flask import Flask,request,jsonify,Blueprint
from flask_cors import CORS, cross_origin
import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.exceptions import OutputParserException
import pandas as pd
import uuid
import chromadb



with open('../config.yaml','r') as file:
    config = yaml.safe_load(file)

api_key = config['groq']['groq_api_key']


cold_email_generation_bp = Blueprint('cold_email_generation_bp',__name__)


class Chain():
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.2-1b-preview")

    def web_data_screping(self,clean_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": clean_text})

        try:

            jsonformate = JsonOutputParser()
            res = jsonformate.parse(res.content)

        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]
    
    def write_email_template(self,job,links):

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Nagendra, a business development executive at A1tech. A1tech is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of A1tech 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase A1tech's portfolio: {link_list}
            Remember you are Mohan, BDE at A1tech. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content
    




