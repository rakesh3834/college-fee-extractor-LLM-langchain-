College Fee Extraction Pipeline (LLM • LangChain • FAISS • Playwright)

A production-grade, AI-powered pipeline for automatically extracting fee structures from university and college websites/PDFs.
This system combines Large Language Models, semantic vector search, and automated scraping to generate clean, structured fee tables with high accuracy.

⭐ Overview

Institutions publish fee details in many formats—PDFs, static HTML, dynamic JavaScript pages, etc. Manual extraction is slow and inconsistent.
This project automates the entire workflow using:

LLM-based structured extraction

Semantic search using FAISS

Automated scraping with Playwright

Robust PDF parsing

The result is a unified, intelligent system that produces standardized fee datasets from any source.

🔍 Key Features
✔ Multi-format Text Extraction

PDFs: via pdfplumber

HTML: via requests + BeautifulSoup

JS-heavy pages: via Playwright

✔ AI-driven Semantic Understanding

Text chunking using LangChain

Embedding generation via OpenAI embeddings

Fee-relevant content retrieved using FAISS vector search

✔ LLM-based Structured Fee Extraction

Automatically extracts fields such as:

Tuition fees

Admission fees

Hostel/mess charges

Caution deposit

Semester/annual fees

Other institutional charges

✔ Clean Output Format

Consolidated fee tables

CSV output with metadata (URL, extraction method, text length, timestamp)

🧠 Architecture & Processing Pipeline
                ┌─────────────────────────┐
                │       Input URL/PDF     │
                └──────────────┬──────────┘
                               ▼
          ┌────────────────────────────────────────┐
          │     Text Extraction Layer              │
          │ Requests → BeautifulSoup / PDF → Playwright │
          └──────────────┬─────────────────────────┘
                         ▼
               ┌───────────────────────┐
               │  Chunking & Embedding │
               │   (LangChain + OpenAI)│
               └──────────────┬────────┘
                              ▼
               ┌────────────────────────┐
               │  Semantic Retrieval     │
               │      (FAISS Index)      │
               └──────────────┬─────────┘
                              ▼
               ┌────────────────────────┐
               │  LLM Fee Table Extraction │
               └──────────────┬─────────┘
                              ▼
              ┌──────────────────────────┐
              │  Consolidated CSV Output │
              └──────────────────────────┘

🏗 Tech Stack
Layer	Technology
LLM Orchestration	LangChain
Embedding Model	OpenAI Embeddings
Vector Search	FAISS
Web Scraping	requests, BeautifulSoup
Browser Automation	Playwright
PDF Parsing	pdfplumber
Output Formatting	Pandas (CSV)
📦 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/college-fee-extractor-LLM.git
cd college-fee-extractor-LLM

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install Playwright (recommended)
playwright install

🔑 Environment Variables

Set your OpenAI API key:

Linux/macOS

export OPENAI_API_KEY="your_api_key_here"


Windows (PowerShell)

setx OPENAI_API_KEY "your_api_key_here"

▶ Usage Example

Inside the script:

result = langchain_fee_extractor(
    "https://home.iitd.ac.in/uploads/ug/24-25/Fee%20Structure24-25.pdf",
    "IIT Delhi",
    custom_query="Extract detailed fee structure including tuition, admission, and hostel charges."
)


Run the main script:

python extractor.py


Output includes:

Structured fee tables

CSV file with timestamp

Metadata about extraction

📁 Project Structure
├── extractor.py                # Main extraction pipeline
├── README.md                   # Project documentation
├── requirements.txt            # Dependency list
└── sample_output/              # Sample output files (optional)

🚀 Future Enhancements

OCR integration for image-based PDFs

REST API deployment

Interactive Streamlit UI

Automated validation of extracted tables

JSON schema export for integrations

🤝 Contributions

Contributions, issues, and feature requests are welcome.
Feel free to open an issue or submit a pull request.
