# workflow.py
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents import (
    cyber_fraud_agent,
    medical_emergency_agent,
    women_safety_agent,
)
from llm_config import get_chat_model
from rag import retrieve_context
from utils import clean_output, normalize_domain, normalize_urgency


class Classification(BaseModel):
    domain: Literal[
        "Women Safety",
        "Cyber Fraud",
        "Medical Emergency",
    ] = Field(description="Primary safety domain")

    urgency: Literal[
        "Critical",
        "High",
        "Moderate",
        "Low",
    ] = Field(description="Urgency level")


class State(TypedDict, total=False):
    user_query: str
    domain: str
    urgency: str
    context: str
    sources: list[dict[str, Any]]
    answer: str


def classify_node(state: State) -> State:
    model = get_chat_model(temperature=0).with_structured_output(
        Classification
    )

    prompt = """
Classify this user message.

Domains:
- Women Safety: threats, harassment, stalking, unsafe surroundings, violence,
  coercion, abduction, or personal danger.
- Cyber Fraud: scams, phishing, impersonation, suspicious payments, malicious
  links, account compromise, or financial fraud.
- Medical Emergency: serious symptoms, injuries, poisoning, breathing issues,
  unconsciousness, severe bleeding, or urgent health concerns.

Urgency:
- Critical: immediate danger, life-threatening symptoms, active violence,
  unresponsiveness, or severe medical warning signs.
- High: urgent threat, active fraud, payment compromise, escalating danger,
  or potentially serious symptoms.
- Moderate: concerning but not clearly immediate.
- Low: general prevention, education, or retrospective questions.

Do not request sensitive information.
"""

    result = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=state["user_query"]),
    ])

    return {
        **state,
        "domain": normalize_domain(result.domain),
        "urgency": normalize_urgency(result.urgency),
    }


def retrieve_node(state: State) -> State:
    context, sources = retrieve_context(
        state["user_query"],
        state["domain"],
        k=4,
    )

    return {
        **state,
        "context": context,
        "sources": sources,
    }


def route_node(state: State) -> Literal[
    "women",
    "cyber",
    "medical",
]:
    return {
        "Women Safety": "women",
        "Cyber Fraud": "cyber",
        "Medical Emergency": "medical",
    }.get(state["domain"], "women")


def women_node(state: State) -> State:
    return {
        **state,
        "answer": clean_output(
            women_safety_agent(
                state["user_query"],
                state["urgency"],
                state["context"],
            )
        ),
    }


def cyber_node(state: State) -> State:
    return {
        **state,
        "answer": clean_output(
            cyber_fraud_agent(
                state["user_query"],
                state["urgency"],
                state["context"],
            )
        ),
    }


def medical_node(state: State) -> State:
    return {
        **state,
        "answer": clean_output(
            medical_emergency_agent(
                state["user_query"],
                state["urgency"],
                state["context"],
            )
        ),
    }


def build_graph():
    graph = StateGraph(State)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("women", women_node)
    graph.add_node("cyber", cyber_node)
    graph.add_node("medical", medical_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_node,
        {
            "women": "women",
            "cyber": "cyber",
            "medical": "medical",
        },
    )

    graph.add_edge("women", END)
    graph.add_edge("cyber", END)
    graph.add_edge("medical", END)

    return graph.compile()


GRAPH = build_graph()


def run_workflow(query):
    if not query or not query.strip():
        raise ValueError("Empty query")

    result = GRAPH.invoke({
        "user_query": query.strip(),
    })

    return {
        "answer": result.get(
            "answer",
            "I could not generate safe guidance.",
        ),
        "domain": result.get("domain", "Unknown"),
        "urgency": result.get("urgency", "Unknown"),
        "sources": result.get("sources", []),
    }