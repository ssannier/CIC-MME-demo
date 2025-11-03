import json
import boto3
import os
import sys

# Set AWS sso profile
os.environ["AWS_PROFILE"] = "CIC-admin"

# Setup clients
session = boto3.Session(profile_name='CIC-admin')
bedrock_runtime = session.client("bedrock-runtime", region_name="us-east-1")
s3vectors = session.client("s3vectors", region_name="us-east-1")

# Configuration
MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
EMBEDDING_DIMENSION = 3072
VECTOR_BUCKET = "cic-s3-bucket-demo"
INDEX_NAME = "cic-demo-3"

def generate_query_embedding(query_text):
    """Generate embedding for user query"""
    request_body = {
        "taskType": "SINGLE_EMBEDDING",
        "singleEmbeddingParams": {
            "embeddingPurpose": "GENERIC_RETRIEVAL",  # Note: RETRIEVAL for queries
            "embeddingDimension": EMBEDDING_DIMENSION,
            "text": {"truncationMode": "END", "value": query_text}
        }
    }
    
    response = bedrock_runtime.invoke_model(
        body=json.dumps(request_body),
        modelId=MODEL_ID,
        contentType="application/json",
    )
    
    response_body = json.loads(response["body"].read())
    return response_body["embeddings"][0]["embedding"]

def search_vectors(query_embedding, top_k=3):
    """Search vector store for most similar embeddings"""
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=INDEX_NAME,
        queryVector={"float32": query_embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    return response["vectors"]

def display_results(results):
    """Print search results"""
    print(f"\n{'='*60}")
    print(f"Found {len(results)} relevant pages:")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        distance = result["distance"]
        similarity = (1 - distance) * 100  # Convert distance to similarity percentage
        
        print(f"{i}. {result['key']}")
        print(f"   Similarity: {similarity:.1f}%")
        print(f"   Source: {metadata.get('source', 'Unknown')}")
        print(f"   Page: {metadata.get('page_number', 'Unknown')} of {metadata.get('total_pages', 'Unknown')}")
        print()

def chatbot():
    """Simple chatbot interface for querying the vector store"""
    print("\n" + "="*60)
    print("Nova Multimodal Vector Search Chatbot")
    print("="*60)
    print(f"\nConnected to:")
    print(f"  Bucket: {VECTOR_BUCKET}")
    print(f"  Index: {INDEX_NAME}")
    print(f"  Embedding Model: {MODEL_ID}")
    print(f"\nType your question to search the PDF document.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            query = input("You: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Skip empty queries
            if not query:
                continue
            
            print("\nSearching...", end=" ")
            
            # Generate embedding for query
            query_embedding = generate_query_embedding(query)
            print("OK")
            
            # Search vector store
            results = search_vectors(query_embedding, top_k=3)
            
            # Display results
            if results:
                display_results(results)
            else:
                print("\nNo results found.")
            
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Please try again.\n")

if __name__ == "__main__":
    try:
        # Verify index exists
        s3vectors.get_index(vectorBucketName=VECTOR_BUCKET, indexName=INDEX_NAME)
        print("Index verified")
        
        # Start chatbot
        chatbot()
        
    except s3vectors.exceptions.NotFoundException:
        print(f"Error: Index '{INDEX_NAME}' not found in bucket '{VECTOR_BUCKET}'")
        print("Please run the embedding script first to create and populate the index.")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        sys.exit(1)