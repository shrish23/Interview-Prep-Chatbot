from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API"))

def generate_interview_summary(chat_history, resume_context, jd_context):
    """
    Generate a summary and feedback of the interview based on chat history,
    resume context, and job description context.

    Args:
        chat_history (list): List of dictionaries with 'role' and 'content' keys.
        resume_context (dict): Extracted features from the resume.
        jd_context (dict): Extracted features from the job description.

    Returns:
        str: The generated summary and feedback.
    """
    try:
        # Format chat history into a readable string
        chat_history_str = ""
        for msg in chat_history:
            role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
            chat_history_str += f"{role}: {msg['content']}\n\n"
        
        # Create system prompt for summary and feedback
        system_prompt = (
            "You are an expert interview coach tasked with analyzing a completed mock interview. "
            "Based on the chat history, resume context, and job description context provided below, "
            "generate a detailed summary of the candidate's performance. Include:\n"
            "1. An overall summary of the interview.\n"
            "2. Specific feedback on which responses were strong, weak, or incorrect.\n"
            "3. Suggestions for improvement tailored to the job description and resume.\n"
            "Use a professional and constructive tone.\n\n"
            f"Chat History:\n{chat_history_str}\n"
            f"Resume Context:\n{resume_context}\n"
            f"Job Description Context:\n{jd_context}\n"
        )
        
        # Prepare messages for the Mistral API
        messages = [ChatMessage(role="system", content=system_prompt)]
        
        # Generate response using Mistral API
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.3,  # Lower temperature for more structured output
            max_tokens=1000   # Allow sufficient length for detailed feedback
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"API Error in summary generation: {e}")
        return "I'm having trouble generating the summary. Please try again."
