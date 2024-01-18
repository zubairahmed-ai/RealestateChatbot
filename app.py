from flask import Flask, render_template, request, jsonify

import os, json, re
from datetime import datetime, timedelta
# os.environ["OPENAI_API_KEY"] = "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"
from typing import Type, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import SimpleMemory

# from typing import Type
# from pydantic import BaseModel, Field
# from langchain.tools import BaseTool
# import golf_utils as gu
from langchain.memory import ConversationBufferWindowMemory, ReadOnlySharedMemory
# from modules import *
from modules import *
import modules.StructuredDateOutput as sdo

from wandb.integration.langchain import WandbTracer
from langchain.agents import AgentType, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain


from enum import Enum
from pydantic import BaseModel, Field

import openai
os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = "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"
os.environ["LANGCHAIN_WANDB_TRACING"] = "false"
os.environ["WANDB_PROJECT"] = "sqldatabasechain"

app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

def check_what_is_empty(user_property_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_property_details.items():
        if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for

def ask_for_info(ask_for = ['location','propertytype', 'rent', 'bedrooms']):

    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below is are some things to ask the user for in a coversation way. you should only ask one question at a time even if you don't get all the info \
        don't ask as a list! Don't greet the user! Don't say Hi.Explain you need to get some info. When specifying property type include these types only - Apartment, House, Condo. If the ask_for list is empty then thank them and ask how you can help them \n\n \
        ### ask_for list: {ask_for}"
    )

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat

def add_non_empty_details(current_details: dict, new_details: dict):
    non_empty_details = {k: v for k, v in new_details.items() if v not in [None, "", 0]}
    current_details.update(non_empty_details)
    return current_details

def filter_response(text_input, property_details):
    chain = create_structured_output_chain(json_schema, llm, prompt, verbose=True)
    # chain = create_tagging_chain_pydantic(PropertyDetails, llm)
    res = chain.run(text_input)
    
    # add filtered info to the
    property_details = add_non_empty_details(property_details,res)
    ask_for = check_what_is_empty(property_details)
    return property_details, ask_for

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat

def _handle_error(error) -> str:
    return str(error)[:50]


cities = GetCities.get_distinct_cities()

agent = None
user_details = None
json_schema = {
        "property": "string",
        "description": "Identifying details of a property for search",
        "type": "object",
        "properties": {
            "location": {"title": "Location", "description": "This is the city and state of the property", "type": "string"},
            "propertytype": {"title": "Property Type", "description": "This is type of the property eg. Apartment, Condo, House", "type": "string"},
            "rent": {
                "title": "Monthly Rent",
                "description": "This is the desired rent of the property",
                "type": "integer",
            },
            "bedrooms": {
                "title": "Bedrooms Number",
                "description": "Bedrooms defined between 1-10 range, convert this type to integer",
                "type": "integer",
            },
        },
        "required": ["location", "propertytype", "rent", "bedrooms"],
    }

user_123_personal_details = None
def define_details():
    global user_123_personal_details
    global user_details
    user_123_personal_details = {'location': '', 'propertytype': '', 'rent': None, 'bedrooms': None}
    user_details = None    


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting information in structured formats. Convert strings and integer into proper formats in the structured data format. Ensure that property type is only one of these types - Apartment, House, Condo else do not extract."),
        ("human", "Use the given format to extract information from the following input: {input}")
    ]
)

def format_text_to_html(output):
    html_table = ""
    pattern = r"^\d+\.\s"

    try:
        # Remove leading numbers and dots
        entries = re.sub(r"\d+\.\s", "", output.strip())

        # Look for patterns "property: value" and split them while preserving commas within values
        properties = re.findall(r"([^:]+): ([^,]+(?:, [^,]+)*)", entries)

        # Start the HTML table
        html_table = "<table>\n<thead>\n<tr>"

        # Define the column headers
        headers = [
            "Property ID", "Location", "Rent", "Type", "Bedrooms", "Renovated", 
            "24/7 Security", "Gym", "Pool", "Pet Friendly", "Parking", "Lease Length",
            "Tour Availability", "School District", "Walk Score", "Bike Score", "Car Reliability"
        ]

        # Add the headers to the table
        for header in headers:
            html_table += f"<th>{header}</th>"

        html_table += "</tr>\n</thead>\n<tbody>\n"

        output = output.split('\n')
        for out in output:
            pairs = [item.strip() for item in out.split(',') if item.strip()]  # Split the string and remove any empty results
            property_dict = {}    
            html_table += "<tr>"
            for pair in pairs:
                if ': ' in pair:  # Check for the delimiter
                    key, value = pair.split(': ', 1)  # Split on the first colon only
                    key = re.sub(pattern, "", key)
                    property_dict[key.strip()] = value.strip()
                    html_table += f"<td>{value.strip()}</td>"                    
            html_table += "</tr>\n"
                    
        
            
            # for value in values[::2]:
            #     # Extract and clean the value            
            #     clean_value = value.strip()

        # Close the table
        html_table += "</tbody>\n</table>"
    except Exception as e:
        print(str(e))
    return html_table

llm = ChatOpenAI(model="gpt-4", temperature=0)

def initialize():
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0, openai_api_key="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"
    )

    property_tools = [
        CheckAvailable.CheckAvailability(),        
    ]
    
    memory=ConversationBufferWindowMemory(k=2)
    memory.output_key = "output"
    prompt_msgs = [
        SystemMessage(
            content="""
            You are also a world class algorithm for understanding the tenant's perspective of searching for properties within desired location, type of desired property, within rent range specified and same or more number of bedrooms than specified.
            Render the results in HTML table as shown below format with column headers from sql column names.

            ### Example of HTML Table Formatting ###
            - <tr><th>Tour Availability</th><th>School District</th><th>Walk Score</th></tr><td>Available</td><td>SLPS</td><td>76</td><tr>
            - <tr><th>Location</th><th>Gym</th><th>Pool</th></tr><td>Long Beach</td><td>Yes</td><td>No</td><tr>
            """
        ),        
        
        HumanMessagePromptTemplate.from_template("{input}"),       

    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)
    
    agent = initialize_agent(property_tools, llm, prompt=prompt, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, return_intermediate_steps=True, memory=memory,handle_parsing_errors=_handle_error, max_execution_time=60) 
    return agent

@app.route("/")
def index():    
    return render_template('Chatbot.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = None
    try:
        msg = request.form["msg"]
    except:
        msg = request.args.get('msg')
    global user_details
    global user_123_personal_details
    if msg == "reset" or msg == "restart":
        define_details()
    # testing only remove lines below
    # user_details = {'location': 'St. Louis', 'propertytype': 'Apartment', 'rent': '2500', 'bedrooms': '1'}
    # return get_Chat_response(json.dumps(user_details))
    # testing only remove lines above

    if user_details is None:    
        user_details, ask_for = filter_response(msg, user_123_personal_details)
    else:
        user_details, ask_for = filter_response(msg, user_details)
    
    if ask_for:
        ai_response = ask_for_info(ask_for)
        if ask_for[0] == 'propertytype':
            ai_response = {"message": ai_response, "options" : ["Apartment", "House", "Condo"]}
        if ask_for[0] == 'location' and len(cities) > 0:
            ai_response = {"message": ai_response, "options" : cities}

        return ai_response
    return get_Chat_response(json.dumps(user_details))

# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_Chat_response(text):    
    response = agent("{\"input\":" + text + "}")
    print(response["intermediate_steps"])
    
    if "Location" in response["output"] or "Property ID" in response["output"]:
        return format_text_to_html(response["output"])

    return response["output"]
    
if __name__ == '__main__':
    agent = initialize()
    define_details()

    app.run(host='0.0.0.0', port=5000)