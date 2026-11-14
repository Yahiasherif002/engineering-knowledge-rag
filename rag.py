import json
import os
import sys
from functools import cache
from pathlib import Path

import psycopg
from fastapi import FastAPI, Query
from openai import OpenAI
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from pypdf import PdfReader

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
READER_DATABASE_URL = os.getenv("READER_DATABASE_URL",
                                "postgresql://agent_reader:agent_reader@localhost:5432/postgres")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = 1536
CHUNK_WORDS, OVERLAP_WORDS = 120, 30
MAX_STEPS, MAX_ROWS = 6, 50

SYSTEM_PROMPT = """You answer questions from plant engineers about their equipment.
Use the tools: search_manuals for manuals and technical specifications, query_equipment for the
equipment register. Use only facts from tool results. After each fact from a manual, cite it as
(file name, p. N). If the tools do not give you the answer, say that you could not find it."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_manuals",
        "description": "Search the equipment manuals and technical specifications. Returns excerpts with file and page.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "query_equipment",
        "description": "Run one read-only PostgreSQL SELECT on the table equipment(tag, model, kind, site, status, "
                       "installed_on, last_service, hours_since_service). kind is pump, transformer or compressor; "
                       "status is running, standby or maintenance. Returns JSON rows.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
    }},
]


@cache
def llm():
    return OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)


def embed(texts):
    response = llm().embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM)
    return [str(item.embedding) for item in response.data]


def chunk(text):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step)]


def ingest(path):
    path = Path(path)
    rows = [(path.name, number, text)
            for number, page in enumerate(PdfReader(path).pages, 1)
            for text in chunk(page.extract_text() or "") if text]
    if not rows:
        return 0
    vectors = embed([text for _, _, text in rows])
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("DELETE FROM chunks WHERE source = %s", (path.name,))
        conn.cursor().executemany(
            "INSERT INTO chunks (source, page, content, embedding) VALUES (%s, %s, %s, %s::vector)",
            [(*row, vector) for row, vector in zip(rows, vectors)])
    return len(rows)


def search(query, limit=5):
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        return conn.execute("""
            WITH terms AS (
                SELECT replace(plainto_tsquery('english', %(query)s)::text, '&', '|')::tsquery AS q
            ), semantic AS (
                SELECT id, row_number() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank
                FROM chunks
                ORDER BY embedding <=> %(vector)s::vector
                LIMIT 20
            ), keyword AS (
                SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
                FROM chunks, terms
                WHERE tsv @@ q
                ORDER BY ts_rank_cd(tsv, q) DESC
                LIMIT 20
            )
            SELECT c.source, c.page, c.content,
                   COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS score
            FROM semantic s
            FULL JOIN keyword k USING (id)
            JOIN chunks c USING (id)
            ORDER BY score DESC
            LIMIT %(limit)s
        """, {"vector": embed([query])[0], "query": query, "limit": limit}).fetchall()


def query_equipment(sql):
    try:
        with psycopg.connect(READER_DATABASE_URL, row_factory=dict_row) as conn:
            rows = conn.execute(sql, prepare=True).fetchmany(MAX_ROWS)
    except psycopg.Error as error:
        return f"SQL error: {error}"
    return json.dumps(rows, default=str)


def ask(question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    sources, queries = [], []
    for _ in range(MAX_STEPS):
        message = llm().chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS).choices[0].message
        if not message.tool_calls:
            return {"answer": message.content, "sources": sources, "queries": queries}
        messages.append({"role": "assistant", "content": message.content,
                         "tool_calls": [call.model_dump(exclude_none=True) for call in message.tool_calls]})
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            if call.function.name == "search_manuals":
                hits = search(args["query"])
                sources += [{"source": h["source"], "page": h["page"]} for h in hits
                            if {"source": h["source"], "page": h["page"]} not in sources]
                result = "\n\n".join(f"[{h['source']}, p. {h['page']}]\n{h['content']}" for h in hits)
            else:
                queries.append(args["sql"])
                result = query_equipment(args["sql"])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    return {"answer": "I could not finish the answer within the step limit.", "sources": sources, "queries": queries}


app = FastAPI(title="Engineering Knowledge Assistant")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/search")
def search_endpoint(q: str = Query(min_length=1, max_length=500), limit: int = Query(5, ge=1, le=20)):
    return search(q, limit)


@app.post("/ask")
def ask_endpoint(body: Question):
    return ask(body.question)


if __name__ == "__main__":
    for pdf in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "docs").glob("*.pdf")):
        print(f"{pdf.name}: {ingest(pdf)} chunks")
