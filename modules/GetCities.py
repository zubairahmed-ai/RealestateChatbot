from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from typing import Type, Optional, Any, Dict, Union, Callable
import json
import openai, os
import sqlite3

def get_distinct_cities():
    # Database file path and table name
    db_path = 'C:\sqlite\Realestateproperties' # Replace with the path to your SQLite database file
    table_name = 'properties' # Replace with the name of your table

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Create a cursor object
    cur = conn.cursor()
    distinct_rows = None

    try:
        # Prepare the SQL query to retrieve all distinct rows from the table
        query = f'SELECT DISTINCT location FROM {table_name}'

        # Execute the query
        cur.execute(query)
        
        # Fetch all distinct row values
        distinct_rows = cur.fetchall()
        if len(distinct_rows)>0:
            distinct_rows = [item[0] for item in distinct_rows]
        # Print the result or process it as needed
        
    finally:
        # Close the cursor and connection
        cur.close()
        conn.close()

    return distinct_rows