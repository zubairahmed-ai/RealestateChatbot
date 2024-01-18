import os
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"

db = SQLDatabase.from_uri("sqlite:///C:\sqlite\golf_chatbot")
llm = OpenAI(temperature=0, verbose=True)
_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
If someone asks for holes availability, they really mean check the Available column in the TeeTime table, a 'Yes' means available.
Use database current date and current time for queries like 'tomorrow', 'today', 'yesterday'
Only use the following tables:

TeeTime

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"

Only use the following tables:

{table_info}

If someone asks for the table foobar, they really mean the employee table.

Question: {input}"""
PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"], template=_DEFAULT_TEMPLATE
)
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True)
print(db_chain.run("What's the availability of 18 holes this week? show me all available date and times"))

while True:
    query = input('Enter your query: ')
    if query == "exit":
        print('ending')        
        break
    else:        
        print(db_chain.run(query))
