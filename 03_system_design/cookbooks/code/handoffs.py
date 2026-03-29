import json
import inspect
from typing import Callable, List, Dict, Any
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# =========================
# Models
# =========================

class Agent(BaseModel):
    name: str
    instructions: str
    tools: List[Callable]
    model: str = "gpt-5.3"


class Response(BaseModel):
    agent: Agent
    messages: List[Dict]


# =========================
# Tool Schema Generator
# =========================

def function_to_schema(func: Callable) -> dict:
    sig = inspect.signature(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        param_type = "string"

        if param.annotation == int:
            param_type = "integer"

        properties[name] = {
            "type": param_type,
            "description": f"{name} parameter"
        }

        if param.default == inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


# =========================
# Tools
# =========================

def escalate_to_human(summary: str):
    """Only call this if explicitly asked to."""
    return f"Escalated to human with summary: {summary}"


def execute_order(product: str, price: int):
    """Execute an order."""
    return f"Order placed for {product} at ${price}"


def look_up_item(search_query: str):
    """Find item ID."""
    return "item_132612938"


def execute_refund(item_id: str, reason: str = "not provided"):
    """Execute refund."""
    return f"Refund issued for {item_id} due to {reason}"


# =========================
# Agent Transfer Functions
# =========================

def transfer_to_sales_agent():
    return sales_agent


def transfer_to_issues_and_repairs():
    return issues_and_repairs_agent


def transfer_back_to_triage():
    return triage_agent


# =========================
# Agents
# =========================

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself briefly. Gather intent and route user."
    ),
    tools=[
        transfer_to_sales_agent,
        transfer_to_issues_and_repairs,
        escalate_to_human
    ],
)

sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc. "
        "Keep replies short. Ask about problems catching roadrunners. "
        "Suggest a crazy product, then give a ridiculous price, then close."
    ),
    tools=[
        execute_order,
        transfer_back_to_triage
    ],
)

issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a support agent. Ask about the issue, suggest fix, "
        "and offer refund if needed."
    ),
    tools=[
        look_up_item,
        execute_refund,
        transfer_back_to_triage
    ],
)


# =========================
# Tool Executor
# =========================

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments or "{}")

    print(f"[Tool Call] {name}({args})")

    return tools_map[name](**args)


# =========================
# Main Agent Loop
# =========================

def run_full_turn(agent: Agent, messages: List[Dict]) -> Response:
    current_agent = agent
    messages = messages.copy()

    for _ in range(10):  # prevent infinite loop

        tool_schemas = [function_to_schema(t) for t in current_agent.tools]
        tools_map = {t.__name__: t for t in current_agent.tools}

        response = client.chat.completions.create(
            model=current_agent.model,
            messages=[
                {"role": "system", "content": current_agent.instructions},
                *messages
            ],
            tools=tool_schemas if tool_schemas else None,
        )

        message = response.choices[0].message
        messages.append(message)

        if message.content:
            print(f"{current_agent.name}: {message.content}")

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            if isinstance(result, Agent):
                current_agent = result
                result = f"Switched to {current_agent.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    return Response(agent=current_agent, messages=messages)


# =========================
# CLI Runner
# =========================

if __name__ == "__main__":
    agent = triage_agent
    messages = []

    while True:
        user_input = input("\nUser: ")
        messages.append({"role": "user", "content": user_input})

        response = run_full_turn(agent, messages)
        agent = response.agent
        messages = response.messages
