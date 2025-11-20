import json
import boto3
import os
from typing import Dict, Any, List

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
s3_client = boto3.client('s3')

# Environment variables
VECTOR_BUCKET = os.environ.get('VECTOR_BUCKET')
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'amazon.nova-lite-v1:0')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')

print(f"Query Handler Lambda initialized - Bucket: {VECTOR_BUCKET}, Model: {BEDROCK_MODEL_ID}")


def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using Bedrock"""
    try:
        response = bedrock_runtime.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response['body'].read())
        return response_body.get('embedding', [])
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []


def search_vectors(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """Search for similar vectors in S3 bucket"""
    try:
        # List objects in vector bucket
        response = s3_client.list_objects_v2(Bucket=VECTOR_BUCKET, MaxKeys=100)
        
        if 'Contents' not in response:
            return []
        
        results = []
        for obj in response['Contents']:
            # Get vector data from S3
            vector_obj = s3_client.get_object(Bucket=VECTOR_BUCKET, Key=obj['Key'])
            vector_data = json.loads(vector_obj['Body'].read())
            
            # Calculate similarity (cosine similarity)
            similarity = calculate_similarity(query_embedding, vector_data.get('embedding', []))
            
            results.append({
                'key': obj['Key'],
                'similarity': similarity,
                'metadata': vector_data.get('metadata', {}),
                'text': vector_data.get('text', '')
            })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        print(f"Error searching vectors: {str(e)}")
        return []


def calculate_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def generate_response(query: str, context: List[Dict]) -> str:
    """Generate response using Bedrock Nova with RAG context"""
    try:
        # Build context from search results
        context_text = "\n\n".join([
            f"Document {i+1} (similarity: {doc['similarity']:.2f}):\n{doc['text']}"
            for i, doc in enumerate(context)
        ])
        
        # Create prompt with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
        
        # Call Bedrock Nova
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "inferenceConfig": {
                    "max_new_tokens": 512,
                    "temperature": 0.7
                }
            })
        )
        
        response_body = json.loads(response['body'].read())
        return response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', 'No response generated')
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"


def handler(event, context):
    """
    Lambda handler for query processing with RAG
    """
    print(f"Received event: {json.dumps(event)}")
    
    try:
        # Parse request body
        if 'body' in event:
            body = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            body = event
        
        query = body.get('query', body.get('question', ''))
        
        if not query:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'No query provided'})
            }
        
        # Generate embedding for query
        query_embedding = get_embedding(query)
        
        if not query_embedding:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Failed to generate query embedding'})
            }
        
        # Search for similar vectors
        search_results = search_vectors(query_embedding, top_k=3)
        
        # Generate response with context
        answer = generate_response(query, search_results)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'query': query,
                'answer': answer,
                'sources': [
                    {
                        'key': doc['key'],
                        'similarity': doc['similarity'],
                        'text_preview': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text']
                    }
                    for doc in search_results
                ],
                'model': BEDROCK_MODEL_ID
            })
        }
    
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'message': 'Internal server error'
            })
        }
