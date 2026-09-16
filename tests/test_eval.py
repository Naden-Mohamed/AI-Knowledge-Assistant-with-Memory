"""
Simple evaluation harness for the assistant's answer quality.

Not a unit test suite — this runs a set of (question, expected_facts) pairs
through the assistant and asks the LLM to judge whether the answer actually
contains the expected facts and is grounded in retrieved context (not
hallucinated). Produces a scored table you can drop straight into your
project's Testing & Evaluation section.

Also verifies memory works: a follow-up question with a pronoun ("its",
"that") that only resolves correctly if the assistant is using chat history.
"""
import json
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings
from src.indexing.vector_store import load_vector_store
from src.pipeline.rag_pipeline import RAGAssistant
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Edit this list to match the actual content of the PDFs/Sheet you indexed.
TEST_CASES = [
    {
        "question": "What is this document about?",
        "expected_facts": "A general, high-level description matching the document's actual topic",
    },

]

# Follow-up pairs to test memory: the second question only makes sense given the first's answer.
MEMORY_TEST = {
    "first_question": "What is this document about?",
    "followup_question": "What are its main limitations?",
}

JUDGE_SYSTEM_PROMPT = (
    "You evaluate a QA assistant's answer against expected facts and the context "
    "it was given. Score 1-5: 5 = accurate and grounded, contains expected facts; "
    "3 = partially correct or vague; 1 = wrong or unsupported by context (hallucinated). "
    "Respond with ONLY JSON: {\"score\": <int>, \"reasoning\": \"<one sentence>\"}"
)


def judge_answer(llm, question, expected_facts, answer, context) -> dict:
    prompt = (
        f"QUESTION: {question}\n\nEXPECTED FACTS: {expected_facts}\n\n"
        f"CONTEXT PROVIDED TO ASSISTANT:\n{context[:2000]}\n\nASSISTANT'S ANSWER:\n{answer}"
    )
    response = llm.invoke([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    text = response.content.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"score": None, "reasoning": f"judge output not parseable: {text[:200]}"}


def run_eval():
    settings = get_settings()
    store = load_vector_store()
    if store is None:
        print("No index found. Run `python main.py --build` first.")
        return

    assistant = RAGAssistant(store)
    judge_llm = ChatOpenAI(
        model=settings.llm_model,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=SecretStr(settings.GEMINI_API_KEY),
    )

    results = []

    print("=== Factual accuracy tests ===")
    for case in TEST_CASES:
        session_id = str(uuid.uuid4())
        result = assistant.ask(case["question"], session_id=session_id)
        context = "\n".join(d.page_content for d in result["source_documents"])
        verdict = judge_answer(judge_llm, case["question"], case["expected_facts"], result["answer"], context)
        results.append({"question": case["question"], "answer": result["answer"], **verdict})
        print(f"Q: {case['question']}\nA: {result['answer']}\nScore: {verdict.get('score')} — {verdict.get('reasoning')}\n")

    print("=== Memory / follow-up test ===")
    session_id = str(uuid.uuid4())
    first = assistant.ask(MEMORY_TEST["first_question"], session_id=session_id)
    followup = assistant.ask(MEMORY_TEST["followup_question"], session_id=session_id)
    print(f"Q1: {MEMORY_TEST['first_question']}\nA1: {first['answer']}\n")
    print(f"Q2 (follow-up): {MEMORY_TEST['followup_question']}")
    print(f"Rewritten standalone question: {followup['standalone_question']}")
    print(f"A2: {followup['answer']}\n")
    memory_worked = MEMORY_TEST["followup_question"].split()[0].lower() not in followup["standalone_question"].lower().split()[:2]
    print(f"Memory check: {'PASS' if memory_worked else 'CHECK MANUALLY'} — pronoun should have been resolved to a specific noun in the rewritten question.\n")

    scored = [r["score"] for r in results if r.get("score") is not None]
    if scored:
        print(f"=== Summary: {len(scored)} cases, average score {sum(scored)/len(scored):.1f}/5 ===")


if __name__ == "__main__":
    run_eval()