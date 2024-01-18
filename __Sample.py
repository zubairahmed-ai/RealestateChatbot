import os
from datetime import datetime, timedelta
os.environ["OPENAI_API_KEY"] = "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"
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
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

os.environ["LANGCHAIN_WANDB_TRACING"] = "false"

os.environ["WANDB_PROJECT"] = "sqldatabasechain"
# db = SQLDatabase.from_uri("sqlite:///C:\sqlite\golf_chatbot",
# include_tables=['TeeTime'], # we include only one table to save tokens in the prompt :)
#     sample_rows_in_table_info=2
# )
# print(db.table_info)
# toolkit = SQLDatabaseToolkit(llm=ChatOpenAI(), db=db)

llm = ChatOpenAI(
    model="gpt-3.5-turbo-0613",
    temperature=0
)

prompt_msgs = [    
    HumanMessage(
        content="If someone asks for holes availability, they really mean check the Available column in the TeeTime table."
    )
]
# prompt = ChatPromptTemplate(messages=prompt_msgs)

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.

If someone asks for holes availability, they really mean check the Available column in the TeeTime table, a 'Yes' means available.
Use database current date and current time for queries like 'tomorrow', 'today', 'yesterday'
Only use the following tables:

TeeTime

Question: {input}"""
# PROMPT = PromptTemplate(
#     input_variables=["input"], template=_DEFAULT_TEMPLATE
# )
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True, memory=SimpleMemory(memories={"date": datetime.now()}),
# )

# def run_db(query: str) -> str:
#     try:
#         result = db_chain(_DEFAULT_TEMPLATE.format(input=query))
#     except:
#         result = "Sorry I couldn't process that, please try again!"
#     # print(result["intermediate_steps"])
#     return result

golf_tools = [
    GolfActivitiesCreate.GolfActivitiesCreate(),
    TeeTimeCreate.TeeTimeCreate(),
    TeeTimeModify.TeeTimeModifyTime(),
    TeeTimeModify.TeeTimeModifyDate(),
    CheckAvailable.CheckAvailability(),
    Tool(name="ConvertFriendlyDateAndTime",
    func=sdo.friendly_date_time,
    description="Use this agent first before other tools to convert friendly dates eg. yesterday, today, tomorrow, next week, next Monday, next Thursday. Current date is " + datetime.now().strftime('%d-%m-%Y') + ". Current time is " + datetime.now().strftime('%H:%M')+ "Don't use this tool to show output to user, instead say 'I don't know what to do with it'"
    ),
    Tool(name="ConvertFriendlyDate",
    func=sdo.friendly_date_time,
    description="Use this agent first before other tools to convert any day, month format eg. 27 July to machine readable dates with relation to current date in a structured formats. Don't use this tool to show output to user, instead say 'I don't know what to do with it'"
    )
]

ALL_TOOLS = golf_tools

memory=ConversationBufferWindowMemory(k=2)
memory.output_key = "output"
# readonlymemory = ReadOnlySharedMemory(memory=memory)

prompt_msgs = [
    SystemMessage(
        content="You are also a world class algorithm for extracting friendly dates eg. yesterday, today, tomorrow to machine readable dates with relation to current date in a structured formats. Current date is " + datetime.now().strftime('%d-%m-%Y')
    ),
    SystemMessage(
        content="You are also a world class algorithm for extracting friendly time eg. 7am, 10pm, 2am to machine readable time with relation to current time in a structured formats. You will also extract month eg. 27, 27 July, 15 August and convert them in relation to current date in a structured formats. You will also refer any dates in relation to current date and current time. Current date is " + datetime.now().strftime('%d-%m-%Y') + ", Current time is " + datetime.now().strftime('%H:%M') + "If Name is not provided, you must ask user and you must not use person name eg. John. If you don't understand anything, just say you don't know. Don't show converted dates to user, say I don't know instead"
    ),
    SystemMessage(
        content="Please don't show converted date and time to user, say 'I don't know what to do with it'"
    ),
    HumanMessage(
        content="Use the given format to extract dates and convert them according to machine readable system date from the following input:"
    ),
    HumanMessage(
        content="Use the given format to extract information from the following input:"
    ),
    HumanMessagePromptTemplate.from_template("{input}"),
    HumanMessage(content="Tips: Make sure to date is system date in the correct format"),

]
prompt = ChatPromptTemplate(messages=prompt_msgs)

def _handle_error(error) -> str:
    return str(error)[:50]
agent = initialize_agent(golf_tools, llm, prompt=prompt, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, return_intermediate_steps=True, memory=memory,handle_parsing_errors=_handle_error, max_execution_time=15)        

# query = "Is 18 holes available this week? show me all available date and times"
# query = "Is there any hole available for today?"
# query = "What's the availability of 18 holes today? show me all available date and times"
# print(agent.run(query))
query = "For James, book a tee time for 9am tomorrow"
# print(agent.run(query))
# query = "10am"
# # query = "need to book 10am on July 27 with golf clubs"
# query = "for tomorrow do you have 8pm available?"
response = agent("{\"input\":" + query + "}")
print(response["output"])
print(' ')
print(' ')
# print(response["intermediate_steps"])

# query = "make that 11am"
# print(agent.run(query))

while True:
    query = input('Enter your query: ')
    if query == "exit":
        print('ending')        
        break
    else:
        response = agent("{\"input\":" + query + "}")
        print(response["output"])
        # print(' ')
        # print(' ')
        # print(response["intermediate_steps"])
