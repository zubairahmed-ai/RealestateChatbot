from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from modules import CheckAvailable as ca

def golf_activities_create_func(name, activity, number_of_persons, date, time, requirements):
    if name == '' or date == '' or time == '' or activity == '':    
        return {
            "To book a Golf activity, please provide a person name, Date, Time and type of activity eg. Golf Lesson, Golf Bay reservation etc."
            }
    insert_activity = ca.run_db("insert activities {Person Name:" + name + ",Number of Persons: " + number_of_persons + ", Date of booking:" +  date + ",Time of Booking:" + time + ",Type of Booking:" + activity + ", requirements:" + requirements + "}")
    return insert_activity


class GolfActivitiesCreateInput(BaseModel):
    """Inputs for the Create Golf Activity Like Golf Lesson or Golf Bay reservation"""
    name: str = Field(description="Name of the person for booking, this is a must")
    activity: str = Field(description="Activity at the Golf Club like Golf lesson")
    number_of_persons: str = Field(description="Number of Persons")
    date: str = Field(description="Date of the Activity")
    time: str = Field(description="Time of the Activity")
    requirements: str = Field(description="Special requirements")


class GolfActivitiesCreate(BaseTool):
    name = "golf_activities_create_func"
    description = """
    Useful when you like to book a Golf lesson, Reserve a facility like a golf bay
    This function is not used to book Tee Time
    You must enter activity type, date, time of the activity, number of persons with special requirements, if any.
    """
    args_schema: Type[BaseModel] = GolfActivitiesCreateInput


    def _run(self, name: str, activity: str, number_of_persons: str, date: str, time: str, requirements: str):
        response = golf_activities_create_func(name, activity, number_of_persons, date, time, requirements)
        return response


    def _arun(self, activity: str, date: str, time: str, requirements: str):
        raise NotImplementedError(f"golf_activities_create_func does not support async")