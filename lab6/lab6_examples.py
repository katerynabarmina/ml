from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

os.environ["OPENAI_API_KEY"] = ""

# Define the model
model = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the workflow
workflow = StateGraph(state_schema=MessagesState)

# Define the model node
def call_model(state: MessagesState):
    chain = prompt | model
    response = chain.invoke(state)
    return {"messages": response}

workflow.add_node("model", call_model)

# Connect the START node to the model node
workflow.add_edge(START, "model")

# Compile the workflow
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Test the workflow
config = {"configurable": {"thread_id": "abc345"}}

# Start conversation
input_messages = [HumanMessage("Hi! I'm Bob.")]

app.invoke({"messages": input_messages}, config)

input_messages.append(HumanMessage("What's my name?"))

app.invoke({"messages": input_messages}, config)

input_messages.append(HumanMessage("Give me a cool name on your taste."))

app.invoke({"messages": input_messages}, config)

input_messages.append(HumanMessage("So what is the apple flavor?"))

app.invoke({"messages": input_messages}, config)

input_messages.append(HumanMessage("Could you add my name on the end of each message?"))

app.invoke({"messages": input_messages}, config)

# Add the request to speak in Ukrainian
input_messages.append(HumanMessage("Please, speak to me in Ukrainian"))

# Get the response in Ukrainian
app.invoke({"messages": input_messages}, config)

input_messages.append(HumanMessage("Яка кількість супутників Юпітера?"))

output = app.invoke({"messages": input_messages}, config)

# Print the entire conversation
print("\nConversation:")
for message in output["messages"]:
    print(message)