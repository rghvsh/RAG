from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(api_key='')

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import tensorflow 
import pandas as pd

# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'who messaged me today'
    'what are my meetings today', 
    ]

# No instruction needed for retrieval passages
passage_prefix = ""
passages = [
    "messages from a Intel",
    "meeting at noon tommorow"
]

# load model with tokenizer
model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)

# get the embeddings
max_length = 4096
query_embeddings = model.encode(queries, instruction=query_prefix, max_length=max_length)

passage_embeddings = model.encode(passages, instruction=passage_prefix, max_length=max_length)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# batch_size=2
# query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32)
# passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32)

scores = (query_embeddings @ passage_embeddings.T) * 100
print(scores.tolist())
#[[77.9402084350586, 0.4248958230018616], [3.757718086242676, 79.60113525390625]]

index_name = "retrieval"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

index = pc.Index(index_name)
index.upsert(
    vectors=[
        {"id": "vec1", "values": [1.0, 1.5]},
        {"id": "vec2", "values": [2.0, 1.0]},
        {"id": "vec3", "values": [0.1, 3.0]},
    ],
    namespace="ns1"
)

index.upsert(
    vectors=[
        {"id": "vec1", "values": [1.0, -2.5]},
        {"id": "vec2", "values": [3.0, -2.0]},
        {"id": "vec3", "values": [0.5, -1.5]},
    ],
    namespace="ns2"
)

query_results1 = index.query(
    namespace="ns1",
    vector=[1.0, 1.5],
    top_k=3,
    include_values=True
)

print(query_results1)

query_results2 = index.query(
    namespace="ns2",
    vector=[1.0,-2.5],
    top_k=3,
    include_values=True
)

print(query_results2)
