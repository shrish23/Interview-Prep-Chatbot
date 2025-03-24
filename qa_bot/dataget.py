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

# model_name = "mistralai/Mistral-Nemo-Instruct-2407"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# llm_model.to(device)
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

# def generate_interview_response(conversation_history, resume_context, jd_context, current_stage):
#     """Generate an interview question or response using the LLM."""
# #     prompt = f"""
# # <s>[INST] You are an expert interviewer conducting a job interview to evaluate a candidate’s fit for a specific role based on their responses. If the candidate provides a job description, use it to guide the interview. If none is provided, assume the role is 'Software Engineer,' with duties including designing, developing, and maintaining software systems, collaborating with teams, and solving technical problems efficiently. Follow this interview process:

# # 1. **Introduction**: Greet the candidate, introduce yourself as the interviewer, and explain that you’ll start with behavioral questions to explore their background, followed by tailored technical and non-technical questions based on their responses and the job description. Begin by asking, 'Can you tell me about yourself?'

# # 2. **Behavioral Questions**: Ask 2-3 open-ended behavioral questions to assess past experiences, teamwork, problem-solving, and work ethic. Examples:
# #    - 'Tell me about a time you faced a challenging work situation and how you resolved it.'
# #    - 'Describe a time you collaborated with a team to meet a goal. What was your role?'
# #    - 'How do you manage task prioritization under tight deadlines?'

# # 3. **Contextual Transition**: Analyze the candidate’s behavioral responses to identify key experiences, projects, or skills. Use these insights to create follow-up questions that explore their expertise and alignment with the job description.

# # 4. **Technical Questions**: Pose 2-3 technical questions tied to the job description and the candidate’s experiences. Test their domain knowledge, problem-solving, and technical skills. Examples:
# #    - If a software project is mentioned: 'Can you describe the system’s architecture and your reasoning behind it?'
# #    - If coding is relevant: 'How would you optimize a slow algorithm in [language/technology they referenced]?'
# #    - If no experience is specified, ask (e.g., for Software Engineer): 'How would you debug a production issue in a distributed system?'

# # 5. **Non-Technical Questions**: Ask 1-2 non-technical questions to evaluate soft skills, cultural fit, or role-specific scenarios, customized to their responses and the job. Examples:
# #    - 'How do you respond to feedback from a stakeholder who disagrees with you?'
# #    - 'How do you keep up with industry trends relevant to this position?'

# # 6. **Dynamic Adaptation**: Adjust your questions dynamically based on the candidate’s answers. If they mention tools, technologies, or methodologies (e.g., Python, Agile, cloud platforms), follow up with:
# #    - 'Can you explain how you used [tool/technology] in that project?'
# #    - 'What obstacles did you encounter with [methodology], and how did you address them?'

# # 7. **Tone and Style**: Adopt a professional, friendly, and engaged tone, showing genuine curiosity about the candidate’s abilities. Use open-ended questions to elicit detailed answers, avoiding yes/no prompts.

# # 8. **Closure**: Conclude by asking, 'Do you have any questions for me about the role or company?' to mimic a real interview.

# # **Important Instructions**:
# # - Generate only the interviewer's question or statement, and stop immediately after. Do not generate a candidate response.
# # - Ensure the output is a single, concise question or statement that aligns with the interview process.
# # - Do not include 'Candidate:' or any candidate response in the output.

# # Tailor all questions to the job description’s requirements (e.g., skills, duties, qualifications) if provided; otherwise, use the default 'Software Engineer' role. Start by greeting the candidate.

# # Resume Context:
# # {resume_context}

# # JD Context:
# # {jd_context}
# # [/INST]
# # """
# #     prompt = f"""
# # <s>[INST] <<SYS>>
# # You are an expert interviewer. Follow these stages strictly in order:
# # 1. **Introduction** → 2. **Behavioral** → 3. **Technical** → 4. **Non-Technical** → 5. **Candidate Q&A** → 6. **Closure**
# # - Never skip, reorder, or combine stages.
# # - Use the job description (if provided) to tailor questions.
# # - Always generate new questions.
# # - Never simulate the candidate’s response.
# # - Do not assume a candidate name unless explicitly provided in the resume or conversation.
# # <</SYS>>

# # ### Current Stage: {current_stage}

# # ### Step 1: Introduction
# # If the current stage is **Introduction**, greet the candidate:
# # "Hello! I'm Shrish, your interviewer. Could you briefly introduce yourself?"
# # - Wait for the candidate’s response before proceeding to Step 2.

# # ### Step 2: Behavioral Questions
# # If the current stage is **Behavioral**, ask two unique questions such as:
# # - "Tell me about a time when you had to resolve a conflict within your team."
# # - "How do you balance multiple high-priority tasks under tight deadlines?"
# # **After each answer**:
# # 1. Provide feedback (e.g., "That’s a good example. How did you handle X challenge?")
# # 2. Ask the next question.

# # ### Step 3: Technical Questions
# # If the current stage is **Technical**, generate two job-specific questions based on the resume and job description, for example:
# # - "Can you describe your experience with [specific technology from resume] in a recent project?"
# # - "How would you approach troubleshooting [relevant technical challenge from JD]?"
# # **After each answer**:
# # 1. Give feedback (e.g., "Interesting solution. How would you optimize it for scalability?")
# # 2. Ask the next question.

# # ### Step 4: Non-Technical Questions
# # If the current stage is **Non-Technical**, ask one original question like:
# # - "How would you explain a complex technical concept to a non-technical audience?"
# # - "Describe a time you had to persuade a reluctant stakeholder."
# # **After the answer:** Provide brief feedback.

# # ### Step 5: Candidate Q&A
# # If the current stage is **Candidate Q&A**, ask:
# # "Do you have any questions about the role or company?"
# # - Use the JD/resume context to provide answers.
# # - If no JD/resume: "I’d need more context, but generally..."
# # - Only move to the next stage after the candidate says, ‘No more questions.’

# # ### Step 6: Closure
# # If the current stage is **Closure**, say:
# # "Great! That concludes our interview for now. Thank you for your time and thoughtful answers."

# # ### Strict Rules:
# # - Never go back to previous stages.
# # - Never simulate candidate responses.
# # - Do not use markdown or lists in responses.
# # - Always generate fresh questions (examples are only for reference).
# # - Each response should be focused only on the current stage.
# # - Wait for a candidate response before transitioning stages.

# # Resume: {resume_context}
# # Job Description: {jd_context} [/INST]
# # """
#     prompt = f"""
#     <s>[INST] You are an expert interviewer conducting a structured job interview. The process has 6 stages: 1. Introduction → 2. Behavioral Questions (2-3 questions) → 3. Technical Questions (2-3 questions) → 4. Non-Technical Questions (1-2 questions) → 5. Candidate Q&A → 6. Closure.

#     ### Rules:
#     - Never skip, reorder, or combine stages.
#     - Use the job description (JD) to tailor questions; if none, assume 'Software Engineer' (designing, developing, maintaining software, collaborating, problem-solving).
#     - Always ask open-ended questions.
#     - Never simulate candidate responses or include 'Candidate:' in output.
#     - Provide feedback after each answer in Behavioral, Technical, and Non-Technical stages.
#     - Do not use markdown or lists in responses.

#     ### Current Stage: {current_stage}

#     ### Resume Context: {resume_context}
#     ### JD Context: {jd_context}

#     ### Conversation History:
#     """
#     for msg in conversation_history:
#         role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
#         prompt += f"{role}: {msg['content']}\n"
#     prompt += """
#     ### Task:
#     Generate the next interviewer's question or statement based on the current stage. Use a chain of thought to reason step-by-step:

#     1. Identify the current stage and its requirements (e.g., Introduction: greet and ask about the candidate; Behavioral: ask 2-3 questions with feedback).
#     2. Review the conversation history to determine the progress within the stage (e.g., how many questions asked, feedback given).
#     3. Use the resume and JD to tailor the question if applicable.
#     4. Ensure the response is a single, open-ended question or statement, with a professional and friendly tone.
#     5. If feedback is required (Behavioral, Technical, Non-Technical), include it before the next question.

#     ### Chain of Thought:
#     - Step 1: The current stage is {current_stage}. What does this stage require?
#     - Step 2: Based on the conversation history, how many questions have been asked in this stage, and is feedback needed?
#     - Step 3: What relevant information from the resume ({resume_context}) or JD ({jd_context}) can I use?
#     - Step 4: What is the next appropriate question or statement?

#     ### Final Response (Interviewer):
#     """

#     # for msg in conversation_history:
#     #     role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
#     #     prompt += f"{role}: {msg['content']}\n"
#     # prompt += "Interviewer:"

#     inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
#     with torch.no_grad():
#         outputs = llm_model.generate(**inputs, max_new_tokens=150, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     response = response.split("Interviewer:")[-1].strip()
#     if "Candidate:" in response:
#         response = response.split("Candidate:")[0].strip()
#     return response

# def generate_interview_response(conversation_history, resume_context, jd_context, current_stage):
    """Generate an interview question or response using Chain of Thought prompting."""
    try:
        if isinstance(resume_context, str):
            resume_data = json.loads(resume_context) if resume_context else {}
        else:
            resume_data = resume_context if resume_context else {}
        name = resume_data.get("name", None)
        experiences = resume_data.get("experiences", None)
    except json.JSONDecodeError:
        name = None
        experiences = None
        print("Error: Invalid resume context JSON")

    try:
        if isinstance(jd_context, str):
            jd_data = json.loads(jd_context) if jd_context else {}
        else:
            jd_data = jd_context if jd_context else {}
        role = jd_data.get("role", None)
        key_responsibilities = jd_data.get("key_responsibilities", None)
    except json.JSONDecodeError:
        role = None
        key_responsibilities = None
        print("Error: Invalid JD context JSON")

    # Prepare context strings for the prompt
    resume_context_str = f"Name: {name}, Experiences: {experiences}" if name or experiences else "No resume data available"
    jd_context_str = f"Role: {role}, Key Responsibilities: {key_responsibilities}" if role or key_responsibilities else "JD features not present"
    prompt = f"""
    <s>[INST] You are an expert interviewer named Shrish conducting a structured job interview with 6 stages: 1. Introduction → 2. Behavioral Questions (2-3 questions) → 3. Technical Questions (2-3 questions) → 4. Non-Technical Questions (1-2 questions) → 5. Candidate Q&A → 6. Closure.

    Rules:
    - Never skip, reorder, or combine stages.
    - The JD context and resume context are provided as JSON objects. Extract the 'role' and 'key_responsibilities' from the JD context JSON.
    - Extract the 'name' and 'experiences' from the resume context JSON.
    - Always ask open-ended questions.
    - Never simulate candidate responses.
    - Provide feedback after each answer in Behavioral, Technical, and Non-Technical stages.
    - Output only the final interviewer's response, without reasoning, markdown, or lists.
    - Do not display the internal reasoning.

    Current Stage: {current_stage}
    Resume Context (JSON): {resume_context_str}
    JD Context (JSON): {jd_context_str}

    Conversation History:
    """
    for msg in conversation_history:
        role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        prompt += f"{role}: {msg['content']}\n"
    prompt += """
    Task: Generate the next interviewer's question or statement for the current stage. Use internal chain-of-thought reasoning to decide, then provide only the final response.

    Internal Reasoning (do not include in output):
    1. What is the current stage, and what does it require?
    2. How many questions have been asked in this stage, based on the conversation history?
    3. Is feedback needed before asking the next question?
    4. Parse the JD context JSON to extract the 'role' and 'key_responsibilities'. If the JSON is empty or missing these fields, Say that JD features not present.
    5. Parse the resume context JSON to extract the 'name' and 'experiences'. If the JSON is empty or missing these fields, use a generic greeting.
    6. What relevant info from the extracted 'role', 'key_responsibilities', 'name', 'experiences', or conversation history can I use?
    7. What is the next question or statement, in a professional and friendly tone?

    Final Response (Interviewer):
    [Write your final interviewer statement here]
    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=150,  # Sufficient for a single response
            temperature=0.5, # Stop at extra newlines, reasoning, or candidate simulation
            # eos_token_id=tokenizer.encode("Internal Reasoning")[0]    # Stop at the end of the response
        )
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Debug: Log the raw response to inspect what the model generates
    

    # clean_response = re.sub(
    #     r"Final Response \(Interviewer\):(.*?)Internal Reasoning:",
    #     "Final Response (Interviewer):",
    #     raw_response,
    #     flags=re.DOTALL
    # )
    # print("Raw Model Output:", clean_response)
    # # Extract the final response
    # if "Final Response (Interviewer):" in clean_response:
    #     response = clean_response.split("Final Response (Interviewer):")[-1].strip()
    # else:
    #     # Fallback: Take the last non-empty line if the separator is missing
    #     response_lines = [line.strip() for line in clean_response.split("\n") if line.strip()]
    #     response = response_lines[-1] if response_lines else "Error: No response generated"

    # response = re.sub(r"(Candidate:|Step|Internal Reasoning)", "", response).strip()

    # # Ensure no candidate simulation or reasoning
    # if any(keyword in response for keyword in ["Candidate:", "Step", "Internal Reasoning"]):
    #     response = "Error: Invalid response detected, please retry."

    # return response
    print("Raw Model Output:", raw_response)
    match = re.search(r"Final Response \(Interviewer\):(.*?)Internal Reasoning:", raw_response, flags=re.DOTALL)
    if match:
        response = match.group(1).strip()
    else:
        # Fallback: extract everything after "Final Response (Interviewer):" if "Internal Reasoning:" is not found
        match = re.search(r"Final Response \(Interviewer\):(.*)", raw_response, flags=re.DOTALL)
        response = match.group(1).strip() if match else "Error: No response generated"

    # Remove any leftover markers (just in case)
    response = re.sub(r"(Candidate:|Step|Internal Reasoning)", "", response).strip()

    if not response:
        response = "Error: No valid response generated, please retry."

    return response

# def generate_chain_of_thought(
#     conversation_history,
#     resume_context,
#     jd_context,
#     current_stage,
#     tokenizer,
#     llm_model,
#     device
# ):
#     """
#     First pass: Generate the chain-of-thought reasoning.
#     This pass does NOT return the final interviewer response—just the reasoning.
#     """

#     # Prepare minimal prompt to retrieve chain-of-thought
#     prompt = f"""
# <s>[INST] You are an expert interviewer named Shrish. 
# You will generate ONLY your chain-of-thought reasoning for the next question or statement.
# Do NOT produce the final question or statement yet.

# Context:
# - Current Stage: {current_stage}
# - Resume: {resume_context}
# - JD: {jd_context}
# - Conversation History:
# """
#     for msg in conversation_history:
#         role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
#         prompt += f"{role}: {msg['content']}\n"

#     prompt += """
# Task: Provide a chain-of-thought describing your reasoning steps. 
# Do not include the final interview question or statement. 
# Do not include "Final Response (Interviewer):" in this pass.

# Chain-of-Thought:
# """

#     inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
#     with torch.no_grad():
#         outputs = llm_model.generate(
#             **inputs,
#             max_new_tokens=150,
#             temperature=0.5,
#         )
#     raw_reasoning = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Raw Reasoning:", raw_reasoning)

#     # We expect the raw_reasoning to contain the model's chain-of-thought.
#     # You might do some cleaning or trimming here if needed.
#     return raw_reasoning.strip()

# def generate_final_answer(
#     chain_of_thought,
#     conversation_history,
#     resume_context,
#     jd_context,
#     current_stage,
#     tokenizer,
#     llm_model,
#     device
# ):
#     """
#     Second pass: Use the chain-of-thought to produce a final interviewer response,
#     WITHOUT revealing that reasoning to the user.
#     """

#     prompt = f"""
# <s>[INST] You are an expert interviewer named Shrish.
# You have the following chain-of-thought from your internal reasoning:
# \"\"\"{chain_of_thought}\"\"\"

# Now produce ONLY the final interviewer response for the next question or statement,
# based on your chain-of-thought and the conversation so far. 
# Follow these rules:
# - Do NOT include chain-of-thought in the output.
# - Do NOT simulate the candidate response.
# - Output only the final interviewer response in plain text (no markdown, no lists).

# Context:
# - Current Stage: {current_stage}
# - Resume: {resume_context}
# - JD: {jd_context}
# - Conversation History:
# """
#     for msg in conversation_history:
#         role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
#         prompt += f"{role}: {msg['content']}\n"

#     prompt += """
# Final Response (Interviewer):
# """

#     inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
#     with torch.no_grad():
#         outputs = llm_model.generate(
#             **inputs,
#             max_new_tokens=200,
#             temperature=0.5,
#         )
#     raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Raw Output from generate_final_answer: ", raw_output)

#     # Now we parse out the final answer
#     # Look for "Final Response (Interviewer):" to isolate the last portion
#     match = re.search(r"Final Response \(Interviewer\):(.*)", raw_output, flags=re.DOTALL)
#     if match:
#         final_answer = match.group(1).strip()
#     else:
#         # Fallback if not found
#         lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
#         final_answer = lines[-1] if lines else "No final answer generated"

#     # Clean up any leftover chain-of-thought references
#     final_answer = re.sub(r"(Candidate:|Step|Internal Reasoning)", "", final_answer).strip()

#     return final_answer

# def generate_interview_response(conversation_history, resume_context, jd_context, current_stage):
#     """
#     A two-pass approach:
#     1. Get chain-of-thought (hidden from user).
#     2. Generate final interviewer response using that chain-of-thought.
#     """
#     # Pass 1: Retrieve chain-of-thought
#     reasoning = generate_chain_of_thought(
#         conversation_history,
#         resume_context,
#         jd_context,
#         current_stage,
#         tokenizer,
#         llm_model,
#         device
#     )

#     # Pass 2: Use chain-of-thought to produce final answer
#     final_response = generate_final_answer(
#         reasoning,
#         conversation_history,
#         resume_context,
#         jd_context,
#         current_stage,
#         tokenizer,
#         llm_model,
#         device
#     )

#     return final_response

# def generate_interview_response(conversation_history, resume_context, jd_context, current_stage):
    """Generate an interview question or response for Mistral-7B-Instruct-v0.2."""
    # Parse resume context
    try:
        if isinstance(resume_context, str):
            resume_data = json.loads(resume_context) if resume_context else {}
        else:
            resume_data = resume_context if resume_context else {}
        name = resume_data.get("name", None)
        experiences = resume_data.get("experiences", None)
    except json.JSONDecodeError:
        name = None
        experiences = None
        print("Error: Invalid resume context JSON")

    # Parse JD context
    try:
        if isinstance(jd_context, str):
            jd_data = json.loads(jd_context) if jd_context else {}
        else:
            jd_data = jd_context if jd_context else {}
        role = jd_data.get("role", None)
        key_responsibilities = jd_data.get("key_responsibilities", None)
    except json.JSONDecodeError:
        role = None
        key_responsibilities = None
        print("Error: Invalid JD context JSON")

    # Format concise context strings
    if experiences and isinstance(experiences, list):
        experiences_str = ", ".join(
            [f"{exp.get('position', 'Unknown')} at {exp.get('company', 'Unknown')} "
             f"({exp.get('start_date', 'Unknown')} - {exp.get('end_date', 'present')})"
             for exp in experiences]  # Limit to top 1 experience
        )
    else:
        experiences_str = "No experiences provided"
    resume_context_str = f"Name: {name}, Experiences: {experiences_str}" if name or experiences else "No resume data available"

    if key_responsibilities and isinstance(key_responsibilities, list):
        responsibilities_str = ", ".join(key_responsibilities)  # Limit to top 2 responsibilities
    else:
        responsibilities_str = "No key responsibilities provided"
    jd_context_str = f"Role: {role}, Key Responsibilities: {responsibilities_str}" if role or key_responsibilities else "JD features not present"

    # Define stage-specific task
    if current_stage == "Introduction":
        task = "Generate a concise introductory statement that greets the candidate by name, mentions the role from JD Context, and asks one open-ended question about themselves and their background. Output only one response and stop immediately. Do not simulate candidate responses or provide feedback."
    elif current_stage == "Behavioral Questions":
        task = "Generate two open-ended behavioral question to assess past experiences, teamwork, or problem-solving, without listing multiple options or simulating responses."
    elif current_stage == "Technical Questions":
        task = "Generate two technical question tied to the job description and candidate’s experiences, focusing on domain knowledge or problem-solving, without listing multiple options or simulating responses."
    elif current_stage == "Non-Technical Questions":
        task = "Generate one non-technical question to evaluate soft skills, cultural fit, or role-specific scenarios, without listing multiple options or simulating responses."
    elif current_stage == "Candidate Q&A":
        task = "Ask, 'Do you have any questions for me about the role or company?' and stop. Do not simulate responses."
    elif current_stage == "Closure":
        task = "Provide a closing statement to conclude the interview, thank the candidate for their time, and mention the next steps. Do not simulate responses."
    else:
        task = "Unknown stage"

    # Construct the prompt
    prompt = f"""
    <s>[INST] You are an expert interviewer named Shrish conducting a structured job interview with 6 stages: 1. Introduction → 2. Behavioral Questions (2-3 questions) → 3. Technical Questions (2-3 questions) → 4. Non-Technical Questions (1-2 questions) → 5. Candidate Q&A → 6. Closure.

    Rules:
    - Never skip, reorder, or combine stages.
    - Use the provided role and key responsibilities from JD Context; if 'JD features not present', proceed without role-specific tailoring.
    - Use the provided name and experiences from Resume Context; if 'No resume data available', use a generic greeting.
    - Always ask one open-ended question unless the stage is Closure or Candidate Q&A.
    - Never simulate candidate responses, provide feedback, or include meta-commentary like 'I could ask...'.
    - Output ONLY one final interviewer's response after 'Final Response (Interviewer):', with no reasoning, markdown, lists, or additional text. Stop immediately after the response.

    Current Stage: {current_stage}
    Task: {task}
    Resume Context: {resume_context_str}
    JD Context: {jd_context_str}

    Conversation History:
    """
    for msg in conversation_history:
        role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        prompt += f"{role}: {msg['content']}\n"
    prompt += """
    Final Response (Interviewer):
    """

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=80, temperature=0.3)
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Debug: Log raw response
    print("Raw Model Output:", raw_response)
    raw_response_lines = raw_response.split("\n")
    print("Raw Response Lines:", raw_response_lines)

    # Extract the final response
    match = re.search(r"Final Response \(Interviewer\):(?:(?!Final Response \(Interviewer\):).)*Final Response \(Interviewer\):\s*(.*)", raw_response, re.DOTALL)
    if match:
        response = match.group(1).strip()
        print("Extracted Response:", response)
        # Take only the first line to avoid trailing content
        response_lines = [line.strip() for line in response.split("\n") if line.strip()]
        print("Response Lines:", response_lines)
        response = response_lines[1] if response_lines else response
    else:
        response = "Error: No response generated"

    # Check for simulated candidate responses or disallowed content
    # simulated_patterns = [r"\bI\b", r"\bme\b", r"\bmy\b", r"\byou\b", r"worked on a project", r"I'm excited"]
    # if any(re.search(pattern, response, re.IGNORECASE) for pattern in simulated_patterns):
    #     print(f"Simulated response detected: {response}")
    #     response = "Error: Invalid response detected, please retry."

    disallowed_keywords = ["candidate:", "internal", "feedback", "I could ask", "perhaps I can ask"]
    if any(keyword.lower() in response.lower() for keyword in disallowed_keywords):
        print(f"Disallowed keyword detected: {response}")
        response = "Error: Invalid response detected, please retry."

    return response

# def generate_interview_response(conversation_history, resume_context, jd_context, current_stage):
#     """Generate an interview question or response for a structured interview process."""
    
#     # Parse Resume Context
#     try:
#         resume_data = json.loads(resume_context) if isinstance(resume_context, str) else resume_context or {}
#         name = resume_data.get("name", "Candidate")
#         experiences = resume_data.get("experiences", [])
#     except json.JSONDecodeError:
#         name, experiences = "Candidate", []
#         print("Error: Invalid resume context JSON")

#     # Parse Job Description Context
#     try:
#         jd_data = json.loads(jd_context) if isinstance(jd_context, str) else jd_context or {}
#         role = jd_data.get("role", "the position")
#         key_responsibilities = jd_data.get("key_responsibilities", [])
#     except json.JSONDecodeError:
#         role, key_responsibilities = "the position", []
#         print("Error: Invalid JD context JSON")

#     # Summarize Contexts
#     experiences_str = ", ".join([
#         f"{exp.get('position', 'Unknown')} at {exp.get('company', 'Unknown')} ({exp.get('start_date', 'Unknown')} - {exp.get('end_date', 'present')})"
#         for exp in experiences
#     ]) if experiences else "No relevant experience found."

#     responsibilities_str = ", ".join(key_responsibilities[:2]) if key_responsibilities else "No key responsibilities provided."

#     resume_context_str = f"Name: {name}, Experiences: {experiences_str}"
#     jd_context_str = f"Role: {role}, Key Responsibilities: {responsibilities_str}"

#     # **Ensure the Interview Moves Forward**
#     if current_stage == "Introduction" and conversation_history:
#         if "tell me about yourself" in conversation_history[-1]["content"].lower():
#             current_stage = "Behavioral Questions"

#     # **Stage-Specific Task Assignment**
#     task_map = {
#         "Introduction": "Ask an introductory question based on the candidate's name and job role.",
#         "Behavioral Questions": "Ask one behavioral question about teamwork, leadership, or problem-solving.",
#         "Technical Questions": "Ask a technical question based on job responsibilities and the candidate’s past projects.",
#         "Non-Technical Questions": "Ask one question about soft skills or company culture fit.",
#         "Candidate Q&A": "Ask if the candidate has any questions for you about the company or role.",
#         "Closure": "Provide a closing statement and thank the candidate for their time."
#     }
#     task = task_map.get(current_stage, "Generate the next question in the structured interview.")

#     # **Build the Improved Prompt**
#     prompt = f"""
#     <s>[INST] You are an expert interviewer conducting a structured job interview in the following stages:
#     1. Introduction → 2. Behavioral Questions → 3. Technical Questions → 4. Non-Technical Questions → 5. Candidate Q&A → 6. Closure.

#     **Rules:**
#     - Move logically between stages; never repeat past questions.
#     - Use the provided job role and responsibilities to tailor questions.
#     - Reference past conversation history to ask relevant follow-ups.
#     - Never simulate candidate responses, provide feedback, or generate multiple questions at once.
#     - Output ONLY one interviewer response after **'Final Response (Interviewer):'** and stop immediately.

#     **Current Stage:** {current_stage}
#     **Task:** {task}
#     **Resume Context:** {resume_context_str}
#     **JD Context:** {jd_context_str}

#     **Conversation History:**
#     """
#     for msg in conversation_history[-5:]:  # Limit to last 5 messages to reduce token usage
#         role = "Interviewer" if msg["role"] == "assistant" else "Candidate"
#         prompt += f"{role}: {msg['content']}\n"

#     prompt += """
#     Final Response (Interviewer):
#     """

#     # **Generate Model Response**
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
#     with torch.no_grad():
#         outputs = llm_model.generate(**inputs, max_new_tokens=250, temperature=0.6)
    
#     raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print("Raw Model Output:", raw_response)

#     # **Extract Only the Final Interview Question**
#     match = re.search(r"Final Response \(Interviewer\):\s*(.*)", raw_response, re.DOTALL)
#     if match:
#         response = match.group(1).strip()
#     else:
#         response = "Error: No valid response generated."

#     # **Prevent Duplicating the Introduction**
#     if "I'm your interviewer" in response or "Can you introduce yourself" in response:
#         if current_stage != "Introduction":
#             response = "Error: Model repeated introduction. Resetting stage."

#     return response

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
            "2. Adapt questions based on the candidate's responses.\n"
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