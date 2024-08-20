import telethon
from telethon.sync import TelegramClient, events 
from telethon.sync import TelegramClient
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

api_id = int(input("Enter telegram api_id: "))
api_hash = input("Enter api_hash: ")
index_name = input("Enter index name: ")
chat = input("Enter chat for which you want to retrieve messages: ")
api_key = input("Enter the api_key for pinecone: ")

pc = Pinecone(api_key= api_key)



if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension= 384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
 
vectors1 = []
metadata= []
with TelegramClient("fetch1", api_id, api_hash) as client:
    for message in client.iter_messages(chat):
        a = message.text + " "+ str(message.date) + ' ' + str(message.media_unread)
        metadata.append(a)
        xq = model.encode(a).tolist()
        vectors1.append(xq)

index = pc.Index(index_name)

for vectorx in range(len(vectors1)):
    index.upsert(
        vectors=[
            {"id": str(vectorx),
              "values": vectors1[vectorx],
              "metadata": {"text": metadata[vectorx] }
            },
    ],
   )
