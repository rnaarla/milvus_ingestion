import json
import hashlib
import os
import typer
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

app = typer.Typer()

# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)


def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def compute_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def connect_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)


def create_collection(collection_name: str):
    if utility.has_collection(collection_name):
        typer.echo(f"Collection '{collection_name}' already exists.")
        return

    fields = [
        FieldSchema(name="hash_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields, description="RAG collection schema")
    collection = Collection(name=collection_name, schema=schema)
    typer.echo(f"Collection '{collection_name}' created.")


@app.command()
def connect(host: str = "localhost", port: str = "19530"):
    """Connect to a Milvus instance."""
    connect_milvus(host, port)
    typer.echo("Connected to Milvus.")


@app.command()
def init_collection(collection: str):
    """Create a collection with standard RAG schema."""
    connect_milvus(os.getenv("MILVUS_HOST", "localhost"), os.getenv("MILVUS_PORT", "19530"))
    create_collection(collection)


@app.command()
def embed(input_file: Path, output_file: Path):
    """Embed and normalize text from JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []
    for doc in data:
        text = doc.get("text", "").strip()
        if not text:
            continue
        embedding = normalize(model.encode(text))
        output.append({
            "hash_id": compute_hash(text),
            "content": text,
            "embedding": embedding.tolist()
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    typer.echo(f"Embedded {len(output)} documents into '{output_file}'")


@app.command()
def insert(collection: str, input_file: Path):
    """Insert embedded records into Milvus."""
    connect_milvus(os.getenv("MILVUS_HOST", "localhost"), os.getenv("MILVUS_PORT", "19530"))
    collection = Collection(collection)
    collection.load()

    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    ids = [r["hash_id"] for r in records]
    contents = [r["content"] for r in records]
    embeddings = [r["embedding"] for r in records]

    collection.insert([ids, contents, embeddings])
    collection.flush()
    typer.echo(f"Inserted {len(ids)} records into '{collection.name}'")


@app.command()
def search(collection: str, query: str, top_k: int = 5, rerank: bool = True):
    """Search for similar content in Milvus and optionally rerank."""
    connect_milvus(os.getenv("MILVUS_HOST", "localhost"), os.getenv("MILVUS_PORT", "19530"))
    collection = Collection(collection)
    collection.load()

    query_embedding = normalize(model.encode(query)).tolist()

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["content"]
    )

    hits = [hit.entity.get("content") for hit in results[0]]

    if rerank:
        pairs = [(query, doc) for doc in hits]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        for i, (doc, score) in enumerate(ranked):
            typer.echo(f"\nRank {i+1} [Score: {score:.4f}]:\n{doc}")
    else:
        for i, doc in enumerate(hits):
            typer.echo(f"\nResult {i+1}:\n{doc}")


@app.command()
def create_index(collection: str):
    """Create an HNSW index on the embedding field."""
    connect_milvus(os.getenv("MILVUS_HOST", "localhost"), os.getenv("MILVUS_PORT", "19530"))
    collection = Collection(collection)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 32, "efConstruction": 200}}
    )
    typer.echo(f"Index created for collection '{collection.name}'")


@app.command()
def batch_insert(collection: str, folder: Path):
    """Batch insert multiple JSON files from a folder."""
    connect_milvus(os.getenv("MILVUS_HOST", "localhost"), os.getenv("MILVUS_PORT", "19530"))
    collection = Collection(collection)
    collection.load()

    for file in folder.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            records = json.load(f)

        ids = [r["hash_id"] for r in records]
        contents = [r["content"] for r in records]
        embeddings = [r["embedding"] for r in records]

        collection.insert([ids, contents, embeddings])
        typer.echo(f"Inserted {len(ids)} records from '{file.name}'")

    collection.flush()
    typer.echo("Batch insertion complete.")


if __name__ == "__main__":
    app()
