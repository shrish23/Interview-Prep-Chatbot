import streamlit as st
import tempfile
import json
import os
import torch

# Import functions from your test.py (adjust module name if needed)
from resume_parsing.resumerag import extract_data, create_collection, upload_json_data
from resume_parsing.databaseconnect import connect_to_database
from qa_bot.dataget import fetch_initial_context, generate_interview_response
from summary_bot.summary import generate_interview_summary

if "current_page" not in st.session_state:
    st.session_state.current_page = "upload"
if "upload_complete" not in st.session_state:
    st.session_state.upload_complete = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "Introduction"
if "question_count" not in st.session_state:
    st.session_state.question_count = 0

# Dummy embedding string creator for vectorization
def embedding_string_creator(data):
    """
    Create a string representation for vectorization.
    Combines resume and job description features into a single string.
    """
    resume_part = data["resume_features"].get("raw_text", " ".join(str(v) for v in data["resume_features"].values()))
    jd_part = data["jd_features"].get("raw_text", " ".join(str(v) for v in data["jd_features"].values()))
    return resume_part + " " + jd_part

# Initialize session state variables
# if 'upload_complete' not in st.session_state:
#     st.session_state.upload_complete = False

if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None

def update_collection_name():
    st.session_state.collection_name = st.session_state.collection_input

def upload_page():
    st.title("Interview Prep Data Extraction & Upload")

    # File uploader for Resume
    resume_file = st.file_uploader("Upload your Resume (PDF/DOC/DOCX)", type=["pdf", "doc", "docx"])

    # Text area for Job Description input
    jd_text = st.text_area("Enter the Job Description", "")

    # Text input for Collection Name
    # collection_name = st.text_input("Enter Collection Name", "interview_data")
    st.text_input(
        "Enter Collection Name",
        value=st.session_state.collection_name,
        key="collection_input",
        on_change=update_collection_name
    )

    # Single button to extract and upload
    if st.button("Extract and Upload"):
        collection_name = st.session_state.collection_name
        if resume_file and jd_text and collection_name:
            # Save uploaded resume file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp:
                tmp.write(resume_file.getvalue())
                resume_path = tmp.name
            
            # Step 1: Extract data
            st.info("Extracting features from resume and job description...")
            extracted_json_str = extract_data(resume_path, jd_text)
            try:
                extracted_data = json.loads(extracted_json_str)
                st.json(extracted_data)  # Display the extracted data
                success = True
            except json.JSONDecodeError:
                st.error("Error parsing extracted data.")
                success = False
            
            # Step 2: Upload data if extraction was successful
            if success:
                st.info("Connecting to database...")
                db = connect_to_database()
                collection = create_collection(db, collection_name)
                
                st.info("Uploading data to collection...")
                upload_json_data(collection, extracted_data, embedding_string_creator)
                
                st.success(f"Data uploaded to collection {collection.full_name}")
                
                # Set session state to indicate upload is complete
                st.session_state.upload_complete = True
                st.session_state.current_page = "interview"
                st.rerun()
        else:
            st.error("Please provide a resume file, job description, and collection name before processing.")

def confirmation_page():
    st.title("Upload Successful")
    st.success("Your data has been successfully extracted and uploaded to the database!")
    st.write("You can now perform additional actions or close this window.")
    
    # Optional: Add a button to return to the upload page
    if st.button("Upload Another File"):
        st.session_state.upload_complete = False
        st.rerun()


def interview_page():
    # Cleaning the CUDA memory
    torch.cuda.empty_cache()
    st.title("Welcome to Interview Bot")

    # Connect to database and retrieve collection
    db = connect_to_database()
    collection = db.get_collection(st.session_state.collection_name)
    resume_context, jd_context = fetch_initial_context(collection)

    # Initialize session states
    defaults = {
        "question_count": 0,
        "chat_history": [],
        "interview_stage": "Introduction"
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    print(f"Current Stage: {st.session_state.interview_stage}")
    print(f"Question Count: {st.session_state.question_count}")

    # Initial system message
    if not st.session_state.chat_history:
        response = generate_interview_response([], resume_context, jd_context, st.session_state.interview_stage)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if user_input := st.chat_input("Your response:"):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.question_count += 1

        print(f"Updated Question Count: {st.session_state.question_count}")

        # Generate and display assistant response
        with st.spinner("Analyzing response..."):
            response = generate_interview_response(
                st.session_state.chat_history,
                resume_context,
                jd_context,
                st.session_state.interview_stage
            )
            
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Update interview stage logic
        stage_transitions = {
            "Introduction": ("Behavioral", 1),
            "Behavioral": ("Technical", 5),
            "Technical": ("Non-Technical", 3),
            "Non-Technical": ("Candidate Q&A", 2),
            "Candidate Q&A": ("Closure", 0)
        }
        
        current_stage = st.session_state.interview_stage
        if current_stage in stage_transitions:
            next_stage, threshold = stage_transitions[current_stage]
            if st.session_state.question_count >= threshold:
                st.session_state.interview_stage = next_stage
                st.session_state.question_count = 0
                print(f"Transitioning to {next_stage} stage")
                if next_stage == "Candidate Q&A":
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Do you have any questions for me?"
                    })

        st.rerun()

    if st.session_state.interview_stage == "Closure":
        if st.button("See Interview Summary"):
            st.session_state.current_page = "summary"
            st.rerun()

def reset_interview():
    st.session_state.chat_history = []
    st.session_state.interview_stage = "Introduction"
    st.session_state.question_count = 0
    st.session_state.current_page = "upload"

def summary_page():
    torch.cuda.empty_cache()
    st.title("Interview Summary and Feedback")
    
    # Connect to database and fetch contexts
    client = connect_to_database()
    collection = client.get_collection(st.session_state.collection_name)
    resume_context, jd_context = fetch_initial_context(collection)
    
    # Display chat history in an expander
    with st.expander("View Chat History"):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Generate and display summary
    st.write("## Interview Summary and Feedback")
    with st.spinner("Generating summary..."):
        summary = generate_interview_summary(st.session_state.chat_history, resume_context, jd_context)
    st.write(summary)
    
    # Option to start a new interview
    if st.button("Start New Interview"):
        reset_interview()
        st.rerun()

# Render the appropriate page based on session state
# if st.session_state.upload_complete:
#     interview_page()
# else:
#     upload_page()
if st.session_state.current_page == "upload":
    upload_page()
elif st.session_state.current_page == "interview":
    interview_page()
elif st.session_state.current_page == "summary":
    summary_page()