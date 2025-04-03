import subprocess
import sys

def run_command(description, command):
    print(f"\n🔹 {description} ...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed at step: {description}")
        sys.exit(result.returncode)
    print(f"✅ Completed: {description}")

if __name__ == "__main__":
    # Step 0: Install Required Python Packages
    run_command(
        "Installing required Python packages",
        "pip install typer sentence-transformers pymilvus cross-encoder tqdm"
    )

    # Step 1: Connect to Milvus
    run_command(
        "Connecting to Milvus",
        "python rag_cli_tool.py connect --host localhost --port 19530"
    )

    # Step 2: Create Collection
    run_command(
        "Creating collection",
        "python rag_cli_tool.py init-collection --collection rag_docs"
    )

    # Step 3: Embed Text File
    run_command(
        "Embedding input text",
        "python rag_cli_tool.py embed --input-file docs.json --output-file vectors.json"
    )

    # Step 4: Insert Embedded Vectors
    run_command(
        "Inserting vectors into Milvus",
        "python rag_cli_tool.py insert --collection rag_docs --input-file vectors.json"
    )

    print("\n🎉 Ingestion pipeline executed successfully.")
    
