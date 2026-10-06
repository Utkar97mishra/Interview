# Top 50 LangChain Interview Questions

A reference guide covering core LangChain concepts, components, and code examples — from models and chains to agents, memory, RAG, and LCEL.

---

## 1. What are the core components of LangChain?

Think of LangChain as a framework to build LLM apps (chatbots, RAG, agents, etc.). LangChain doesn't create its own models — it connects to them.

| Component | Description |
|---|---|
| **Models** | LLMs / Chat models (OpenAI, Anthropic, local models, etc.) |
| **Prompts** | Text sent to the model, often built with templated variables (e.g. `{user_input}`) |
| **Chains** | A fixed pipeline: Prompt → Model → (optional) Output Parser → Final Answer |
| **Memory** | Remembers previous messages and injects past chat history into the prompt |
| **Tools** | Functions the model can call (search, query a DB, call an API, math, etc.) — mainly used by agents |
| **Agents** | "Smart controllers" that decide what to do next, which tool to use, and when to respond |
| **Documents / Text Splitters / Indexes** | Documents = source text (PDFs, web pages); splitters break large docs into chunks; indexes/vector stores enable RAG search |
| **Output Parsers** | Convert raw model text into structured data (JSON, Python dict, custom objects) |

### Minimal example

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Model
model = ChatOpenAI(model="gpt-4o-mini")

# 2. Prompt template
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms for a beginner."
)

# 3. Output parser (just returns a string)
parser = StrOutputParser()

# 4. Chain = Prompt | Model | Parser
chain = prompt | model | parser

# 5. Run the chain
answer = chain.invoke({"topic": "LangChain"})
print(answer)
```

This example uses **models, prompts, chains, and an output parser** — four core components.

---

## 2. What is the difference between a chain and an agent in LangChain?

| Feature | Chain | Agent |
|---|---|---|
| Flow | Fixed | Dynamic |
| Who decides next step? | You (developer) | LLM (agent "brain") |
| Tools | Usually none or fixed | Can choose from multiple tools |
| Complexity | Simple to build, predictable | Powerful but more complex |

### Chain example (fixed steps)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template(
    "Translate this sentence into Hindi:\n\n{sentence}"
)

chain = prompt | model | StrOutputParser()
print(chain.invoke({"sentence": "I love learning LangChain."}))
```

This chain will always just translate text — nothing more, nothing less.

### Agent example (dynamic tool use)

```python
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# 1. Define a tool
@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b

tools = [add]

# 2. Model
model = ChatOpenAI(model="gpt-4o-mini")

# 3. Agent prompt (simplified)
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant that can use tools to solve problems."
)

# 4. Create agent and executor
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 5. Ask a question; agent decides whether to call add()
result = agent_executor.invoke({"input": "What is 12 + 30?"})
print(result["output"])
```

Here, the agent reads the question, decides to call the `add` tool, gets the result, and then replies.

---

## 3. How does LangChain handle memory?

Memory in LangChain remembers previous interactions and passes them back to the model.

**Without memory:**
