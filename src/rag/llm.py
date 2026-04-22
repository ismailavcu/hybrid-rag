import ollama

def llm(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context_text}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    #print("response: \n", response)
    return response.message.content