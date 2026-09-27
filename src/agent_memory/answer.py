"""Answering from a memory context, and judging the answer against the gold one
(model arm). The judge is told the question type because the right standard differs:
an off-by-one day count is fine for a temporal question, the *updated* value is the
only right one for a knowledge-update question, and "I don't know" is the correct
answer to an unanswerable question."""

from __future__ import annotations

import re

from agent_memory.context import Context
from agent_memory.datasets import Question
from agent_memory.llm import CHAT_MODEL, LLM

ANSWER_PROMPT = """You are an assistant with a long-term memory of earlier conversations.
Below is what your memory retrieved for this question. Each session is headed by the
date it took place; use those dates to work out when things happened. If the memory
does not contain the answer, say you do not know rather than guessing.

Memory:
{context}

Today is {now}.
Question: {question}
Answer in one or two sentences."""

_RULES = {
    "temporal": "Dates or durations that are off by at most one day still count as correct.",
    "knowledge-update": (
        "The question asks for the current value. If the response gives the updated value "
        "it is correct, even if it also mentions the earlier one; if it gives only the "
        "outdated value it is wrong."
    ),
    "preference": (
        "The gold answer is a rubric describing what a good personalised response would "
        "use. The response is correct if it draws on the user's stated preference."
    ),
    "abstention": (
        "This question cannot be answered from the conversation. The response is correct "
        "only if it says the information is not available or it does not know."
    ),
}

JUDGE_PROMPT = """Decide whether a response answers a question correctly, given the gold
answer. {rule}

Question: {question}
Gold answer: {gold}
Response: {response}

Is the response correct? Reply with one word: yes or no."""


def answer_prompt(q: Question, ctx: Context) -> str:
    return ANSWER_PROMPT.format(
        context=ctx.render() or "(nothing retrieved)",
        now=f"{q.now:%Y-%m-%d (%A)}",
        question=q.question,
    )


def judge_prompt(q: Question, response: str) -> str:
    rule = _RULES.get(q.qtype, "Paraphrases and extra correct detail are fine.")
    return JUDGE_PROMPT.format(rule=rule, question=q.question, gold=q.answer, response=response)


def parse_verdict(reply: str) -> bool | None:
    """True/False from the judge's reply; None if it said neither."""
    m = re.search(r"\b(yes|no)\b", reply.lower())
    return None if m is None else m.group(1) == "yes"


def answer_and_judge(
    q: Question, ctx: Context, client: LLM, model: str = CHAT_MODEL, judge: str | None = None
) -> dict[str, object]:
    """Answer from the context with `model`, then grade with `judge` (default: the same
    model - self-judging, a known bias; pass another model to avoid it). `correct` is
    None when the judge's reply is neither yes nor no; callers count it as wrong."""
    response = client.generate(model, answer_prompt(q, ctx), {"num_predict": 200})
    reply = client.generate(judge or model, judge_prompt(q, response), {"num_predict": 5})
    return {"response": response, "judge_reply": reply, "correct": parse_verdict(reply)}
