CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given the conversation history and a follow-up question, rewrite the "
    "follow-up as a standalone question that contains all necessary context. "
    "Do NOT answer the question — only rewrite it. If it is already standalone, "
    "return it unchanged."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a knowledge assistant. Answer the user's question using ONLY the "
    "provided context below. If the context doesn't contain the answer, say so "
    "plainly rather than guessing.\n\nContext:\n{context}"
)
