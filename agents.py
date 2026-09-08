# agents.py
from langchain_core.messages import HumanMessage, SystemMessage
from llm_config import get_chat_model


BASE_PROMPT = """
You are KAVACHX, a cautious emergency-assistance assistant.

Rules:
- Be concise, calm, practical, and action-oriented.
- Never request or repeat passwords, OTPs, PINs, card numbers, account numbers,
  security answers, or banking credentials.
- Never claim to have contacted authorities, police, hospitals, banks, or anyone.
- Never invent emergency numbers, organizations, policies, sources, or facts.
- Use only the supplied reference context.
- Recommend local professional or emergency help when appropriate.
- Do not diagnose or prescribe.
"""


def run_general_agent(domain, urgency, query, context):
    model = get_chat_model(temperature=0.1)

    prompt = f"""
{BASE_PROMPT}

Domain: {domain}
Urgency: {urgency}

Reference context:
{context}

Respond with:
1. The most important immediate action.
2. Up to five concise numbered steps.
3. A short safety note.
4. A clear recommendation for professional help when needed.
"""

    result = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ])

    return result.content


def women_safety_agent(query, urgency, context):
    return run_general_agent(
        "Women Safety",
        urgency,
        query,
        context,
    )


def cyber_fraud_agent(query, urgency, context):
    return run_general_agent(
        "Cyber Fraud",
        urgency,
        query,
        context,
    )


def medical_emergency_agent(query, urgency, context):
    model = get_chat_model(temperature=0.0)

    prompt = f"""
{BASE_PROMPT}

You are handling a possible medical emergency.

Medical rules:
- Do not diagnose.
- Do not prescribe medication or dosages.
- Focus on observable symptoms and immediate first-response actions.
- Recommend professional emergency help for severe, sudden, worsening, or unclear
  symptoms.
- If the person is unresponsive or not breathing normally, advise contacting local
  emergency services immediately and following dispatcher instructions.
- Never tell the user to wait for the chatbot.

Urgency: {urgency}

Reference context:
{context}

Return:
1. Immediate action.
2. Up to four concise safety steps.
3. Clear professional-help recommendation.
"""

    result = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=query),
    ])

    return result.content