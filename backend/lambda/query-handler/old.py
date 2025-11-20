"""Lambda handler: receive a prompt, embed it, store embedding in S3,
perform a simple vector search over embeddings stored in S3, then
call Bedrock to generate a response using the retrieved context.

Notes:
- Configure the following environment variables in the Lambda config:
  - VECTOR_BUCKET: S3 bucket name where embeddings are stored (prefix: `vectors/`)
  - EMBED_MODEL: Bedrock embedding model id (default: 'amazon.titan-embed-001')
  - GEN_MODEL: Bedrock text-generation model id (e.g. 'anthropic.claude' or a Titan text model)
  - TOP_K: number of similar docs to retrieve (default: 3)

This implementation is intentionally small-scale: it downloads all vectors from S3
and performs an in-memory cosine similarity search. For large collections,
use a dedicated vector DB or OpenSearch with k-NN indexing.
"""

import os
import json
import uuid
import math
import heapq
import logging
from typing import List, Dict, Any, Optional

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clients (use default session / role from Lambda execution environment)
bedrock = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")

# Config from env
VECTOR_BUCKET = os.environ.get("VECTOR_BUCKET", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "amazon.titan-embed-001")
GEN_MODEL = os.environ.get("GEN_MODEL", "amazon.titan-instruct-001")
TOP_K = int(os.environ.get("TOP_K", "3"))
VECTORS_PREFIX = os.environ.get("VECTORS_PREFIX", "vectors/")


def read_bedrock_response(resp) -> Any:
	"""Read bedrock-runtime invoke_model response body and parse JSON."""
	# response['body'] is a streaming body; read bytes and decode
	raw = resp["body"].read()
	try:
		return json.loads(raw.decode("utf-8"))
	except Exception:
		# fallback: return raw string
		return raw.decode("utf-8")


def find_first_embedding(obj) -> Optional[List[float]]:
	"""Recursively search for the first list of floats in parsed response."""
	if isinstance(obj, list):
		# if list of numbers
		if obj and all(isinstance(x, (int, float)) for x in obj):
			return [float(x) for x in obj]
		# otherwise search elements
		for item in obj:
			e = find_first_embedding(item)
			if e:
				return e
	elif isinstance(obj, dict):
		for v in obj.values():
			e = find_first_embedding(v)
			if e:
				return e
	return None


def get_embedding(text: str) -> List[float]:
	"""Call Bedrock to produce an embedding for `text`.

	The exact response shape depends on the model. We try to parse the
	numeric vector out of the returned JSON.
	"""
	payload = {"input": text}
	logger.info("Invoking Bedrock embed model %s", EMBED_MODEL)
	resp = bedrock.invoke_model(
		modelId=EMBED_MODEL,
		contentType="application/json",
		accept="application/json",
		body=json.dumps(payload),
	)
	parsed = read_bedrock_response(resp)
	embedding = find_first_embedding(parsed)
	if not embedding:
		raise ValueError(f"Could not find embedding in Bedrock response: {parsed}")
	return embedding


def store_embedding_s3(item_id: str, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
	key = f"{VECTORS_PREFIX}{item_id}.json"
	payload = {"id": item_id, "text": text, "embedding": embedding, "metadata": metadata or {}}
	logger.info("Storing embedding to s3://%s/%s", VECTOR_BUCKET, key)
	s3.put_object(Bucket=VECTOR_BUCKET, Key=key, Body=json.dumps(payload).encode("utf-8"))


def list_vector_objects() -> List[str]:
	objs = []
	kwargs = {"Bucket": VECTOR_BUCKET, "Prefix": VECTORS_PREFIX}
	while True:
		resp = s3.list_objects_v2(**kwargs)
		contents = resp.get("Contents", [])
		for c in contents:
			objs.append(c["Key"])
		if resp.get("IsTruncated"):
			kwargs["ContinuationToken"] = resp.get("NextContinuationToken")
		else:
			break
	return objs


def load_embedding_from_key(key: str) -> Dict[str, Any]:
	resp = s3.get_object(Bucket=VECTOR_BUCKET, Key=key)
	body = resp["Body"].read().decode("utf-8")
	return json.loads(body)


def cosine_similarity(a: List[float], b: List[float]) -> float:
	dot = sum(x * y for x, y in zip(a, b))
	na = math.sqrt(sum(x * x for x in a))
	nb = math.sqrt(sum(y * y for y in b))
	if na == 0 or nb == 0:
		return 0.0
	return dot / (na * nb)


def search_similar(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
	keys = list_vector_objects()
	heap = []  # min-heap of (score, item)
	for key in keys:
		try:
			obj = load_embedding_from_key(key)
			emb = obj.get("embedding")
			if not emb:
				continue
			score = cosine_similarity(query_embedding, emb)
			if len(heap) < top_k:
				heapq.heappush(heap, (score, obj))
			else:
				if score > heap[0][0]:
					heapq.heapreplace(heap, (score, obj))
		except Exception as e:
			logger.warning("Failed to load/score %s: %s", key, e)
	# return sorted desc
	return [item for score, item in sorted(heap, key=lambda x: x[0], reverse=True)]


def call_bedrock_generation(prompt: str, max_tokens: int = 1024) -> str:
	payload = {"input": prompt}
	logger.info("Invoking Bedrock generative model %s", GEN_MODEL)
	resp = bedrock.invoke_model(
		modelId=GEN_MODEL,
		contentType="application/json",
		accept="application/json",
		body=json.dumps(payload),
	)
	parsed = read_bedrock_response(resp)
	# The generation text may be under different keys depending on model
	if isinstance(parsed, dict):
		# try common keys
		for k in ("outputText", "generated_text", "text", "content"):
			if k in parsed:
				return parsed[k]
		# try find the first string
		if isinstance(parsed.get("results"), list) and parsed["results"]:
			r = parsed["results"][0]
			if isinstance(r, dict) and "output" in r:
				return r["output"]
		return json.dumps(parsed)
	return str(parsed)


def build_prompt_with_context(user_prompt: str, contexts: List[Dict[str, Any]]) -> str:
	pieces = ["Use the following context to answer the question. If it's not relevant, say you don't know.\n"]
	for i, c in enumerate(contexts):
		pieces.append(f"Context {i+1} (score not shown):\n{c.get('text')}\n---\n")
	pieces.append(f"Question:\n{user_prompt}\n")
	return "\n".join(pieces)


def lambda_handler(event, context):
	logger.info("Event: %s", event)
	body = event.get("body")
	if isinstance(body, str):
		try:
			body = json.loads(body)
		except Exception:
			body = {"prompt": body}
	prompt = (body or {}).get("prompt")
	if not prompt:
		return {"statusCode": 400, "body": json.dumps({"error": "missing prompt"})}

	# 1) embed user prompt
	try:
		embedding = get_embedding(prompt)
	except Exception as e:
		logger.exception("Embedding failed")
		return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

	# 2) store embedding in S3
	item_id = str(uuid.uuid4())
	try:
		store_embedding_s3(item_id, prompt, embedding, metadata={"source": "user_prompt"})
	except Exception as e:
		logger.warning("Failed to store embedding: %s", e)

	# 3) search similar embeddings
	try:
		contexts = search_similar(embedding, top_k=TOP_K)
	except Exception as e:
		logger.exception("Search failed")
		contexts = []

	# 4) format prompt and call generation model
	combined_prompt = build_prompt_with_context(prompt, contexts)
	try:
		answer = call_bedrock_generation(combined_prompt)
	except Exception as e:
		logger.exception("Generation failed")
		return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

	return {"statusCode": 200, "body": json.dumps({"answer": answer, "id": item_id})}


if __name__ == "__main__":
	# quick local test (requires AWS creds in env)
	evt = {"body": json.dumps({"prompt": "What's the weather like?"})}
	print(lambda_handler(evt, None))


