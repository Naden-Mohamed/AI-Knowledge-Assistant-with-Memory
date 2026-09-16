from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ..config import get_settings

# Swap this dict for a Redis/SQLite-backed store if you need persistence across runs.
_session_store: dict[str, BaseChatMessageHistory] = {}


class SummaryBufferHistory(InMemoryChatMessageHistory):
    def __init__(self, llm: ChatOpenAI, trigger_turns: int):
        super().__init__()
        self._llm = llm
        self._trigger_turns = trigger_turns
        self._summary: str | None = None

    def add_message(self, message) -> None:
        super().add_message(message)
        # a "turn" is one human+AI pair, so trigger on ~2x the turn count
        if len(self.messages) >= self._trigger_turns * 2:
            self._compress()

    def _compress(self) -> None:
        """Summarizes everything except the most recent 2 turns, replacing the
        old verbatim messages with one system message holding the summary."""
        keep_recent = 4  # last 2 human+AI turns stay verbatim
        to_summarize = self.messages[:-keep_recent]
        recent = self.messages[-keep_recent:]
        if not to_summarize:
            return

        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in to_summarize
        )
        prior_summary = f"Prior summary: {self._summary}\n\n" if self._summary else ""
        prompt = (
            f"{prior_summary}Summarize the following conversation turns concisely, "
            f"preserving any facts, names, or decisions the user stated that later "
            f"questions might depend on:\n\n{transcript}"
        )
        response = self._llm.invoke([HumanMessage(content=prompt)])
        self._summary = str(response.content)

        self.clear()
        self.add_message(
            SystemMessage(content=f"Summary of earlier conversation: {self._summary}")
        )
        for m in recent:
            self.add_message(m)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Returns (creating if needed) the chat history for a session_id.
    This is the function RunnableWithMessageHistory calls to fetch/store memory."""
    settings = get_settings()
    if session_id not in _session_store:
        if settings.memory_mode == "summary_buffer":
            llm = ChatOpenAI(
                model=settings.llm_model,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=SecretStr(settings.GEMINI_API_KEY),
            )
            _session_store[session_id] = SummaryBufferHistory(
                llm, settings.summary_trigger_turns
            )
        else:
            _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def clear_session(session_id: str) -> None:
    if session_id in _session_store:
        _session_store[session_id].clear()
