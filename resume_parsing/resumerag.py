# from pypdf import PdfReader
import json
import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from astrapy import Database, Collection, DataAPIClient
from astrapy.constants import VectorMetric
from astrapy.info import CollectionVectorServiceOptions
import pypdfium2 as pdfium
from pytesseract import *
from PIL import Image
from io import BytesIO
import re


# Load environment variables (store HuggingFace API key in .env)
load_dotenv()
pytesseract.tesseract_cmd = "C:\\Softwares\\tesseract.exe"
HF_API_KEY = os.getenv("HF_API_KEY")
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to(device)

def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    print("Converted PDF to images")
    return final_images

# 2. Extract text from images via pytesseract


def extract_text_from_img(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    print("Extracted text from images")

    return "\n".join(image_content)


def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract

def extract_json_from_text(generated_text):
    # json_start = generated_text.find("{")
    # json_end = generated_text.rfind("}") + 1
    # if json_start != -1 and json_end != -1:
    #     json_str = generated_text[json_start:json_end]
    #     print(json_str)
    #     try:
    #         return json.loads(json_str)
    #     except json.JSONDecodeError:
    #         return {"error": "Invalid JSON format"}
    # return {"error": "No JSON found in output"}
    inst_end = generated_text.find("[/INST]")
    if inst_end == -1:
        print("Error: No [/INST] tag found in the generated text")
        return {"error": "No [/INST] tag found"}
    
    # Extract the text after [/INST]
    response_text = generated_text[inst_end + len("[/INST]"):]
    
    # Find the first '{' and the last '}' in the response text
    json_start = response_text.find("{")
    json_end = response_text.rfind("}") + 1
    
    if json_start == -1 or json_end == -1:
        print("Error: No valid JSON object found in the response text")
        return {"error": "No JSON found in output"}
    
    # Extract the JSON substring
    json_str = response_text[json_start:json_end]
    
    # Clean the JSON string
    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
    json_str = re.sub(r'\s+', ' ', json_str).strip()          # Normalize whitespace
    
    # Print the cleaned string for debugging
    # print("Cleaned JSON string:", json_str)
    
    try:
        # Parse the cleaned JSON string
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"error": "Invalid JSON format"}

def extract_jd_features(jd_text):
    """Extract key features from job description using HuggingFace LLM"""
    print("Extracting JD features...")
    
    prompt = f"""
    <s>[INST] Extract the following information from the job description and output only a valid JSON object:
    {{
        "role": "",
        "company": "",
        "industry": "",
        "required_skills": [],
        "required_experience": "",
        "education_requirements": "",
        "certifications": [],
        "key_responsibilities": [],
        "salary_range": "",
        "location": "",
        "Extra Requirements": []
    }}
    Do not include any additional text, explanations, or commentary outside the JSON.

    Job Description:
    {jd_text}
    [/INST]
    """
    
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=5000, num_return_sequences=1, do_sample=False)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to parse the generated text as JSON
    # Extract JSON from the generated text
    print("Extracted JD features")
    return extract_json_from_text(generated_text)
    
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    print("Extracting resume data...")

    data = extract_content_from_url(file_path)
    
    prompt = f"""
    <s>[INST] Extract the following information from the resume and output only a valid JSON object::
    {{
        "name": "",
        "contact_number": "",
        "email": "",
        "location": "",
        "summary": "",
        "experiences": [],
        "education": [],
        "projects": [],
        "skills": [],
        "certifications": []
    }}
    Do not include any additional text, explanations, or commentary outside the JSON.

    Resume data:
    {data}
    [/INST]
    """

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=5000, num_return_sequences=1, do_sample=False)

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Attempt to parse the generated text as JSON
    # Extract JSON from the generated text
    print("Extracted resume data")
    return extract_json_from_text(generated_text)
    


def extract_data(resume_path, jd_text=None):
    """Extract features from both resume and job description"""
    resume_features = extract_text_from_pdf(resume_path) if resume_path else {}

    # Parse job description
    jd_features = extract_jd_features(jd_text) if jd_text else {}

    return json.dumps({
        "resume_features": resume_features,
        "jd_features": jd_features
    }, indent=2)

def create_collection(database: Database, collection_name: str) -> Collection:
    """
    Creates a collection in the specified database with vectorization enabled.

    Args:
        database (Database): The instantiated object that represents the database where the collection will be created.
        collection_name (str): The name of the collection to create.

    Returns:
        Collection: The created collection.
    """
    client = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"])
    database = client.get_database(os.environ["ASTRA_DB_API_ENDPOINT"])
    print(f"* Database: {database.info().name}\n")
    collection = database.create_collection(
        name=collection_name,
        metric=VectorMetric.COSINE,
        service=CollectionVectorServiceOptions(
            provider="jinaAI",
            model_name="jina-embeddings-v3",
            authentication={
                "providerKey": "sensei_embeddings",
            },
        ),
        check_exists=True,
    )
    print(f"Created collection {collection.full_name}")

    return collection

def upload_json_data(
    collection: Collection,
    json_data: str,
    embedding_string_creator: callable,
) -> None:
    """
     Uploads data from a file containing a JSON array to the specified collection.
     For each piece of data, a $vectorize field is added. The $vectorize value is
     a string from which vector embeddings will be generated.

    Args:
        collection (Collection): The instantiated object that represents the collection to upload data to.
        resume: file from streamlit
        jd: job description
        embedding_string_creator (callable): A function to create the string for which vector embeddings will be generated.
    """
    # Read the JSON file and parse it into a JSON array.
    # json_data = json.loads(extract_data(resume, jd))
    documents = [{
        **json_data,
        "$vectorize": embedding_string_creator(json_data),
    }]

    # Upload the documents to the collection.
    collection.insert_many(documents)