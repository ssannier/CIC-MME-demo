import json
import boto3
import sys
import os
import base64
from pdf2image import convert_from_path
from io import BytesIO

# Set AWS credentials path
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/Users/sreeramsreedhar/Documents/CIC/nova-test/.aws/credentials"
os.environ["AWS_CONFIG_FILE"] = "/Users/sreeramsreedhar/Documents/CIC/nova-test/.aws/config"

# Setup clients
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
s3vectors = boto3.client("s3vectors", region_name="us-east-1")

# Configuration
MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIMENSION = 3072
VECTOR_BUCKET = "cic-s3-bucket-demo"
INDEX_NAME = "cic-demo-3"
PDF_PATH = "/Users/sreeramsreedhar/Documents/CIC/nova-test/CSE355_TMs.pdf"

# Check/create index
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

# Convert PDF pages to images
print(f"\nConverting PDF to images: {PDF_PATH}")
try:
    # Convert PDF pages to images (handles text, images, diagrams, etc.)
    images = convert_from_path(PDF_PATH, dpi=150)
    print(f"PDF converted successfully: {len(images)} pages")
except Exception as e:
    print(f"Error converting PDF: {e}")
    sys.exit(1)

# Generate embeddings for each page as an image
print(f"\nGenerating embeddings using model: {MODEL_ID}")
vectors_to_store = []

for i, page_image in enumerate(images):
    print(f"Processing page {i+1}/{len(images)}...", end=" ")
    
    # Convert PIL Image to base64
    buffered = BytesIO()
    page_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_INDEX",
            "embeddingDimension": EMBEDDING_DIMENSION,
            "image": {"format": "png", "source": {"bytes": img_base64}},
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
            "key": f"{os.path.basename(PDF_PATH).replace('.pdf', '')}:page_{i+1}",
            "data": {"float32": embedding},
            "metadata": {
                "type": "page_image",
                "source": os.path.basename(PDF_PATH),
                "page_number": str(i+1),
                "total_pages": str(len(images))
            }
        })
        print("✓")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

print(f"\nSuccessfully generated {len(vectors_to_store)} embeddings")

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





