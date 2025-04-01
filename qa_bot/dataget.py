from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import torch
import json
import re
import os

# model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

mistral_client = MistralClient(api_key=os.environ.get("MISTRAL_API"))

# def question_generation(context)
def text_to_vector(text):
    embeddings = model.encode(text)
    print("Embeddings :", embeddings)
    return embeddings.tolist()

# Write a function to perform a vector search in the Astra database
def vector_search(collection, query_vector, top_k=5):
    """Perform a vector search in the Astra database"""
    # Validate the query vector
    if not isinstance(query_vector, list) or not all(isinstance(x, float) for x in query_vector):
        raise ValueError("query_vector must be a list of floats")
    
    try:
        # Perform the search
        results = collection.find(
            sort={"$vector": query_vector},
            limit=top_k
        )
        return list(results)
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []
    
def fetch_initial_context(collection):
    """Automatically fetch resume and JD context from the database."""
    # Define a default query vector (e.g., a neutral vector or use a predefined key)
    default_query = "resume and job description context"  # Adjust based on your data
    query_vector = text_to_vector(default_query)
    
    # Perform vector search to retrieve relevant documents
    results = vector_search(collection, query_vector, top_k=1)
    
    # Extract resume and JD context from the first result
    if results:
        first_doc = results[0]
        resume_context = first_doc.get("resume_features", {})
        jd_context = first_doc.get("jd_features", {})
        return resume_context, jd_context
    return "No resume data available", "No JD data available"


def generate_interview_response(chat_history, resume_context, jd_context, current_stage):
    """Generate interview responses using Mistral's Chat API"""
    try:
        # Create system prompt based on interview stage
        system_prompt = (
            f"You are a professional interview coach conducting a {current_stage} stage interview.\n"
            "Your task is to generate relevant questions and responses based on the provided context.\n\n"
            "Context:\n"
            f"- Resume Features: {resume_context}\n"
            f"- Job Description Features: {jd_context}\n\n"
            "Rules:\n"
            "1. Ask one question at a time.\n"
            "2. Ask follow-up questions that build directly on the candidate's previous responses to explore their answers in more depth.\n"
            "3. Maintain a professional tone.\n"
            "4. For the 'Technical' stage, focus on specific technologies mentioned in the resume and job description.\n"
            "   - If data structures are mentioned, ask questions about arrays, linked lists, trees, graphs, etc.\n"
            "     - Example: 'Can you explain the difference between a stack and a queue?'\n"
            "   - If SQL is mentioned, ask about database design, queries, or optimization.\n"
            "     - Example: 'How would you optimize a slow SQL query?'\n"
            "   - If Spark is mentioned, ask about data processing, RDDs, or DataFrames.\n"
            "     - Example: 'Can you explain the difference between RDDs and DataFrames in Spark?'\n"
            "   - If Hadoop is mentioned, ask about HDFS, MapReduce, or cluster management.\n"
            "     - Example: 'How do you handle data replication in HDFS?'\n"
            "   - If Data Science is mentioned, ask about machine learning models, data cleaning, or statistical analysis.\n"
            "     - Example: 'Can you describe a time when you used a machine learning model to solve a problem?'\n"
            "   - If Python is mentioned, ask about libraries, frameworks, or coding practices.\n"
            "     - Example: 'What is your experience with Python libraries like NumPy or Pandas?'\n"
            "   - If cloud services (AWS, GCP, Azure) are mentioned, ask about specific services, architecture, or deployment.\n"
            "     - Example: 'How would you design a scalable architecture using AWS services?'\n"
            "   - If Data Engineering is mentioned, ask about ETL processes, data warehousing, or pipeline design.\n"
            "     - Example: 'Can you describe your experience with designing and implementing ETL pipelines?'\n"
            "5. For the 'Non-Technical' stage, ask about soft skills, cultural fit, or role-specific scenarios.\n"
            "6. During the 'Candidate Q&A' stage, allow the candidate to ask questions.\n"
            "7. Always respond based on the user's last input.\n"
        )
        
        # Prepare message history
        messages = [ChatMessage(role="system", content=system_prompt)]
        
        # Add previous conversation context
        for msg in chat_history:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append(ChatMessage(role=role, content=msg["content"]))
        
        # Ensure the last message is from the user
        if chat_history and chat_history[-1]["role"] != "user":
            raise ValueError("The last message in chat_history must be from the user.")
        
        temperature = 0.5 if current_stage == "Technical" else 0.7
        
        # Generate response using Mistral API
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=messages,
            temperature=temperature,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"API Error: {e}")
        return "I'm having trouble generating a response. Please try again."

def test_inp(query, collection):
    # Perform a vector search using the input query
    query_vector = text_to_vector(query, model)
    results = vector_search(collection, query_vector)
    return results