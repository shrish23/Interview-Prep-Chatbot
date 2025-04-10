
<br/>
<div align="center">
<a href="https://github.com/ShaanCoding/ReadME-Generator">
<img src="https://i.ibb.co/DDHJ42wt/poster-for-Interview-Prep-AI-project.jpg" alt="Logo" width="80" height="80">
</a>
<h3 align="center">Interview Prep AI</h3>
<p align="center">
The Job You Want Starts With the Practice You Need


  


</p>
</div>

## About The Project

![Product Screenshot](https://picsum.photos/1920/1080)

Interview Prep AI is an innovative tool designed to help job seekers prepare for interviews by simulating a real interview experience. Leveraging natural language processing (NLP) and machine learning, this project generates tailored interview questions based on a user's resume and the provided job description. Beyond question generation, it offers constructive feedback on responses, enabling users to refine their interview skills in a realistic and supportive environment. Whether you're preparing for a technical role or a general position, Interview Prep AI adapts to your background and the job you're targeting, making it an invaluable companion for interview preparation.

**Key Features**
- Personalized Question Generation: Upload your resume and a job description to receive interview questions customized to your experience and the role's requirements.
- Realistic Interview Simulation: Engage in a mock interview where the AI dynamically adjusts its questions based on your answers, mimicking a real-world interview.
- Actionable Feedback: Get detailed feedback on your responses, with suggestions to improve clarity, depth, and relevance.
- User-Friendly Interface: Powered by Streamlit, the tool offers an intuitive web-based experience for seamless interaction.
### Built With

The Libraries and Frameworks used for the project:

- [Streamlit](https://docs.streamlit.io/)
- [Hugging Face](https://huggingface.co/models)
- [Mistral AI](https://docs.mistral.ai/)
- [Astra DB](https://astra.datastax.com/)
- [OCR](https://github.com/h/pytesseract)
- [Jina Embeddings](https://jina.ai/)
- [Pytorch](https://pytorch.org/)
### Prerequisites

Ensure you have the following tools and dependencies installed before proceeding:

- **Python 3.8+**
  ```sh
  # Install Python (example for Ubuntu; adjust for your OS)
  sudo apt update
  sudo apt install python3.8 python3-pip python3-venv -y
  ```
### Installation

_To get started with the project on your local machine/server._

1. To set up your own API keys:
     - Create an account on AstraDB datastax and create a vector database named 'Sensei' and get the API endpoints and the Token (by clicking on the Generate Token).
     - Create an account in Jina AI and get a free API key.
     - Go to 'Integrations' in the AstraDB and select Jina AEmbeddings, and paste the Jina AI API key with the correct name of the Database.
    - Go to Mistral(https://console.mistral.ai/) and create an account to get your free API key.
    - Go to Hugging Face and get your API key.
2. Clone the repo
   ```sh
   git clone https://github.com/shrish23/Interview-Prep-Chatbot
   ```
3. Install the required packages
   ```sh
   pip install -r requirements.txt
   ```
4. Create a .env file in your workspace folder and paste the API keys and endpoints under the following names:
   ```sh
    ASTRA_DB_API_ENDPOINT="YOUR ENDPOINT"
    ASTRA_DB_APPLICATION_TOKEN="YOUR TOKEN"
    JINA_AI_API_KEY="YOUR JINA API KEY"
    HF_API_KEY="YOUR HUGGINGFACE API KEY"
    MISTRAL_API="YOUR MISTRAL API KEY"
   ```
## Usage

You can start the project by using the command:
```sh
streamlit run app.py
```
## License

Distributed under the MIT License. See [MIT License](https://opensource.org/licenses/MIT) for more information.
