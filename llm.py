"""
llm.py
------
Handles all communication with the Groq LLM API (Llama 3).
Produces a structured 4-section fraud analysis report.
"""

import streamlit as st

try:
    from groq import Groq
    _groq_available = True
except ImportError:
    _groq_available = False


def _get_client():
    if not _groq_available:
        return None
    try:
        return Groq(api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
        return None


SYSTEM_PROMPT = """You are a financial fraud detection expert specialising in NLP analysis
of transaction narratives and SEC filings. Your role is to provide concise,
explainable fraud risk assessments for compliance teams.

When given a transaction or filing text, respond in exactly this format
(use ** for bold section headers):

**Summary**: One sentence describing what the text is about.

**Fraud signals**: List 2-4 specific phrases or patterns that raise concern,
or note 'None identified' if the text appears clean. Be specific about what
each signal means in a fraud context.

**Risk reasoning**: 2-3 sentences explaining your overall risk assessment and why.

**Recommended action**: One clear, concrete next step
(e.g. 'Escalate to fraud team', 'Flag for compliance review',
'No action required — routine transaction', etc.).

Be direct and professional. Avoid generic statements.
Focus on what is unique about this particular text."""


def get_llm_analysis(text: str, fraud_prob: float, detected_terms: list) -> str:
    """
    Calls Groq Llama 3 and returns a structured fraud analysis string.
    Falls back gracefully if the API is unavailable.
    """
    client = _get_client()

    if client is None:
        return (
            "⚠️ **AI analysis unavailable** — Groq API key not configured.\n\n"
            "Add your key to `.streamlit/secrets.toml`:\n"
            "```\nGROQ_API_KEY = 'your-key-here'\n```\n\n"
            "Get a free key at: https://console.groq.com"
        )

    user_message = (
        f"Analyse this transaction / filing text for fraud risk:\n\n"
        f"\"{text}\"\n\n"
        f"ML model risk score: {int(fraud_prob * 100)}/100\n"
        f"Detected fraud keywords: "
        f"{', '.join(detected_terms) if detected_terms else 'none'}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=700,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ]
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ AI analysis error: {str(e)}"