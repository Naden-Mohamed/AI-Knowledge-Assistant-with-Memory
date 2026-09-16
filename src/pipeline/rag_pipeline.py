from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document

from ..config import get_settings
from ..memory.conversation_memory import get_session_history
from ..prompts import CONTEXTUALIZE_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT
from pydantic import SecretStr

class RAGAssistant:
    def __init__(self, vector_store):
        settings = get_settings()
        self.retriever = vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})
        self.llm = ChatOpenAI(
            model="gemini-3.6-flash",  # or "gemini-2.5-pro" — see note below
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=SecretStr(settings.GEMINI_API_KEY),
        )

        self._contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        self._contextualize_chain = self._contextualize_prompt | self.llm | StrOutputParser()

        self._answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])

        core_chain = RunnableLambda(self._run)
        self.conversational_chain = RunnableWithMessageHistory(
            core_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",  # only the answer text goes into memory, not source_documents
        )

    def _run(self, inputs: dict) -> dict:
        question = inputs["input"]
        history = inputs.get("history", [])

        standalone_question = (
            self._contextualize_chain.invoke({"input": question, "history": history})
            if history else question
        )

        docs: list[Document] = self.retriever.invoke(standalone_question)
        context = "\n\n".join(
            f"[{d.metadata.get('file_name', d.metadata.get('source_type', 'source'))}] {d.page_content}"
            for d in docs
        )

        answer_chain = self._answer_prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"input": question, "history": history, "context": context})

        return {"answer": answer, "source_documents": docs, "standalone_question": standalone_question}

    def ask(self, question: str, session_id: str) -> dict:
        """Ask a question within a given conversation session. Memory is handled automatically per session_id."""
        return self.conversational_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )