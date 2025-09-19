import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Document = [
    {"text": "GeeksforGeeks (GFG) is one of the most popular online platforms for computer science education. It is widely used by students, developers, and professionals to learn programming, practice coding problems, and prepare for technical interviews."},
    {"text": "GFG Practice Portal allows users to solve coding problems of varying difficulty. Includes daily coding challenges (Problem of the Day).Supports multiple programming languages like C, C++, Java, and Python."},
    {"text": "Offers both free and paid courses in areas such as DSA, Competitive Programming, Web Development, Machine Learning, and System Design. Provides live and self-paced training programs."}
]

def embed_text(text):
    resp= client.models.embed_content(
        model="models/embedding-001",
        contents=text
    )
    return np.array(resp.embeddings[0].values)

DOC_EMBEDDINGS = [embed_text(d["text"]) for d in Document]

def search(query,k=1):
    qvec = embed_text(query)
    sims = [cosine_similarity([qvec], [dv]) for dv in DOC_EMBEDDINGS]
    top_idx=np.argsort(sims)[::-1][:k]
    return [Document[int(i)] for i in top_idx]

print("SMARTED CHATBOT(type 'exit' to quit)\n")
while True:
    query = input("you: ")
    if query.lower()=='exit':
        break
    
    hits=search(query,k=1)
    context = hits[0]["text"]
    
    prompt=f"Answer the question using this info:\n{context}\n\nQuestion: {query}\nAnswer:"
    resp=client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    print("BOT:",resp.text, "\n")