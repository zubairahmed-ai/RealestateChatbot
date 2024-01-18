from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from typing import Type, Optional, Any, Dict, Union, Callable
import json
import openai, os
os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = "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"

db = SQLDatabase.from_uri("sqlite:///C:\sqlite\Realestateproperties",
include_tables=['properties'], # we include only one table to save tokens in the prompt :)
    sample_rows_in_table_info=2
)

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.

If someone asks for location availability including type of property, rent and number of bedrooms they really mean check the Available column in the Properties table with same type of property, same specified or below rent and same specified or more number of bedrooms. Make sure that the same or more number of Bedrooms are also returned, show the complete property details in your response not just the total. 

Only use the following tables:
Properties

Question: {input}"""
# PROMPT = PromptTemplate(
#     input_variables=["input"], template=_DEFAULT_TEMPLATE
# )
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)



def run_db(query: str) -> str:
    try:
        result = db_chain(query)        
    except:
        result = "Sorry I couldn't process that, please try again!"

    return result

def check_availability_func(location='', rent='', propertytype='', bedrooms=''):
    if location == '' and rent == '' and type == '' and bedrooms:
        return {'Sorry, please provide a date to check'}
        
    json = """
    Check Properties on given filters only and give complete property details in response. Location can't be specific so use a Like statement in SQL query generation instead of Equal. Rent can be equal or less than specified, Bedrooms can be same or more than specified. Always get all table columns. Do not summarize the results. Always present result in the following output format
    ### Output Format Example ###
    1. Property ID: 131, Location: St. Louis, MO, Rent: 1300, Type: Apartment, Bedrooms: 1, Renovated: Yes, 24/7 Security: Yes, Gym: No, Pool: No, Pet Friendly: Yes, Parking: Indoor, Lease Length: 12 months, Tour Availability: Available, School District: SLPS, Walk Score: 70, Bike Score: 60, Car Reliability: 65
    """
    json += "{Location:" + location + ","
    json += "Rent:" + rent + ","
    json += "Type:" + propertytype + ","
    json += "Bedrooms:" + bedrooms + "}"
    
    return run_db(query=json)

class CheckAvailabilityInput(BaseModel):
    """Records to identifying which type of holes and courses are available
     along with their date and time. Identifying information about a friendly date and time, converted to machine readable format."""
    location: Optional[str] = Field(description="This is the city and state of the property")
    propertytype: Optional[str] = Field(description="This is type of the property eg. Apartment, Condo, House")
    rent: Optional[str] = Field(description="This is the desired rent of the property")
    bedrooms: Optional[str] = Field(description="This is the desired number of bedrooms, can be more than that")    

class CheckAvailability(BaseTool):
    name = "check_availability_func"
    description = """
    Useful when you like to check the availability of houses, apartments, condons in a given location within the desired rent and same or bedrooms       
    """
    args_schema: Type[BaseModel] = CheckAvailabilityInput
    def _run(self, location:str='', rent:str='', propertytype:str='', bedrooms:str=''):
        response = check_availability_func(location, rent, propertytype, bedrooms)
        return response
    
    def _arun(self, location:str='', rent:str='', propertytype:str='', bedrooms:str=''):
        raise NotImplementedError(f"tee_times_booking_func does not support async")