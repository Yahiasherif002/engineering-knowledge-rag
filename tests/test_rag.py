import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageFunctionToolCall

import rag

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / "docs").glob("*.pdf"))
QUESTIONS = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()]


def fake_vector(text):
    vector = [0.0] * rag.EMBED_DIM
    for word in text.lower().split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % rag.EMBED_DIM] += 1.0
    return vector


def fake_llm(replies):
    def embeddings(model, input, dimensions):
        return SimpleNamespace(data=[SimpleNamespace(embedding=fake_vector(text)) for text in input])

    def chat(model, messages, tools):
        return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])

    return SimpleNamespace(embeddings=SimpleNamespace(create=embeddings),
                           chat=SimpleNamespace(completions=SimpleNamespace(create=chat)))


def tool_call(id, name, **args):
    return ChatCompletionMessageFunctionToolCall(id=id, type="function",
                                                 function={"name": name, "arguments": json.dumps(args)})


def recall_at_5():
    hits = [any((h["source"], h["page"]) == (q["source"], q["page"]) for h in rag.search(q["question"]))
            for q in QUESTIONS]
    return sum(hits) / len(hits)


@pytest.fixture
def replies(monkeypatch):
    queue = []
    monkeypatch.setattr(rag, "llm", lambda: fake_llm(queue))
    for pdf in DOCS:
        assert rag.ingest(pdf) > 0
    return queue


def test_chunk_overlaps():
    words = [f"w{i}" for i in range(200)]
    chunks = rag.chunk(" ".join(words))
    assert [c.split()[0] for c in chunks] == ["w0", "w90"]
    assert chunks[-1].split()[-1] == "w199"


def test_ingest_replaces_old_chunks(replies):
    count = "SELECT count(*) FROM chunks"
    with psycopg.connect(rag.DATABASE_URL) as conn:
        before = conn.execute(count).fetchone()
        rag.ingest(DOCS[0])
        assert conn.execute(count).fetchone() == before


def test_hybrid_search_finds_pages(replies):
    top = rag.search("What torque do I use for the HX-200 impeller nut?")[0]
    assert (top["source"], top["page"]) == ("hx200_pump_manual.pdf", 4)
    assert recall_at_5() >= 0.9


def test_equipment_tool_is_read_only():
    assert json.loads(rag.query_equipment("SELECT count(*) AS n FROM equipment")) == [{"n": 13}]
    assert "read-only" in rag.query_equipment("UPDATE equipment SET status = 'running'")
    assert "permission denied" in rag.query_equipment("SELECT * FROM chunks")
    assert "permission denied" in rag.query_equipment("SELECT pg_read_file('/etc/passwd')")
    assert rag.query_equipment("SELECT * FROM nope").startswith("SQL error")
    assert "multiple commands" in rag.query_equipment("SET statement_timeout = 0; SELECT pg_sleep(10)")
    assert "statement timeout" in rag.query_equipment("SELECT pg_sleep(10)")


def test_agent_uses_both_tools(replies):
    sql = "SELECT tag FROM equipment WHERE model = 'HX-200' AND hours_since_service >= 4000 ORDER BY tag"
    replies += [
        ChatCompletionMessage(role="assistant", tool_calls=[
            tool_call("1", "search_manuals", query="HX-200 service interval"),
            tool_call("2", "query_equipment", sql=sql),
        ]),
        ChatCompletionMessage(role="assistant", content="P-101 and P-104 are due (hx200_pump_manual.pdf, p. 4)."),
    ]
    response = TestClient(rag.app).post("/ask", json={"question": "Which HX-200 pumps are due for service?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("P-101 and P-104")
    assert {"source": "hx200_pump_manual.pdf", "page": 4} in body["sources"]
    assert body["queries"] == [sql]
    assert json.loads(rag.query_equipment(sql)) == [{"tag": "P-101"}, {"tag": "P-104"}]


def test_api_rejects_bad_input():
    client = TestClient(rag.app)
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "pump", "limit": 100}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY for real embeddings")
def test_retrieval_eval_with_real_embeddings():
    for pdf in DOCS:
        rag.ingest(pdf)
    score = recall_at_5()
    print(f"recall@5 = {score:.2f} on {len(QUESTIONS)} questions")
    assert score >= 0.9
