import json
import boto3
import sys
import os
from pypdf import PdfReader
import time

# Set AWS credentials path
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/Users/sreeramsreedhar/Documents/CIC/nova-test/.aws/credentials"
os.environ["AWS_CONFIG_FILE"] = "/Users/sreeramsreedhar/Documents/CIC/nova-test/.aws/config"

# Setup clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")


MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIMENSION = 3072
VECTOR_BUCKET = "test-s3-multimodal-1"
INDEX_NAME = "test1"
PDF_PATH = "/Users/sreeramsreedhar/Documents/CIC/nova-test/CSE355_TMs.pdf"
CHUNK_SIZE = 1000  # Characters per chunk


try:
    s3vectors.get_index(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME)
    print(f"Index '{INDEX_NAME}' exists")
except s3vectors.exceptions.NotFoundException:
    print(f"Creating index '{INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
    s3vectors.create_index(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        dataType="float32",
        distanceMetric="cosine"
    )
    print(f"Index '{INDEX_NAME}' created")
except Exception as e:
    print(f"Error with index: {e}")
    sys.exit(1)

# Read and extract text from PDF
print(f"\nReading PDF: {PDF_PATH}")
try:
    reader = PdfReader(PDF_PATH)
    total_pages = len(reader.pages)
    print(f"PDF loaded successfully: {total_pages} pages")

    # Extract all text from PDF
    full_text = ""
    for i, page in enumerate(reader.pages):
        full_text += page.extract_text()

    print(f"Extracted {len(full_text)} characters from PDF")
except Exception as e:
    print(f"Error reading PDF: {e}")
    sys.exit(1)

# Chunk the text
def chunk_text(text, chunk_size):
    """Split text into chunks of approximately chunk_size characters"""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

chunks = chunk_text(full_text, CHUNK_SIZE)
print(f"Split text into {len(chunks)} chunks")

# Generate embeddings for each chunk
print(f"\nGenerating embeddings using model: {MODEL_ID}")
vectors_to_store = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...", end=" ")

    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": EMBEDDING_DIMENSION,
            "text": {"truncationMode": "END", "value": chunk},
        },
    }

    try:
        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body),
            modelId=MODEL_ID,
            contentType="application/json",
        )

        response_body = json.loads(response["body"].read())
        embedding = response_body["embeddings"][0]["embedding"]

        vectors_to_store.append({
            "key": f"{os.path.basename(PDF_PATH).replace('.pdf', '')}:chunk_{i}",
            "data": {"float32": embedding},
            "metadata": {
                "type": "text",
                "source": os.path.basename(PDF_PATH),
                "chunk_index": str(i),
                "total_chunks": str(len(chunks))
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

print(f"Successfully generated {len(vectors_to_store)} embeddings")

# Store all embeddings to S3 Vectors
print(f"\nStoring {len(vectors_to_store)} vectors to S3 Vectors")
try:
    s3vectors.put_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        vectors=vectors_to_store
    )
    print(f"Successfully stored all vectors")
    print(f"Bucket: {VECTOR_BUCKET}")
    print(f"Index: {INDEX_NAME}")
    print(f"Total vectors: {len(vectors_to_store)}")
except Exception as e:
    print(f"Error storing vectors: {e}")
    sys.exit(1)

print("done!")

