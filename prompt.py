SYSTEM_PROMPT = """
You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not contained in the context, say:
"I couldn't find that information in the provided documents."
Be concise and accurate.
"""

def build_prompt(history, context, question):
    return f"""
    {SYSTEM_PROMPT}

    Conversation History:
    ---------------------
    {history}

    Context:
    ---------------------
    {context}

    ---------------------

    Question:
    {question}

    Answer:
    """