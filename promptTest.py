# flake8: noqa
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""

class Tool:
    def __init__(self, name:str, desc:str):
        self.name = name
        self.description = desc

tools = [Tool(str(i), f"Tool {i}") for i in range(10)]

tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
tool_names = ", ".join([tool.name for tool in tools])
format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
template = "\n\n".join([PREFIX, tool_strings, format_instructions, SUFFIX])

print(template)