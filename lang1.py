import requests
from bs4 import BeautifulSoup
import pdfplumber
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Agent imports removed for compatibility
import json
import pandas as pd
from datetime import datetime
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Playwright not available. Install with: pip install playwright")

# Initialize OpenAI API
os.environ["OPENAI_API_KEY"] = "API KEY HERE"

# Initialize embeddings and LLM with OpenAI
embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0.1, max_tokens=1024)

def extract_with_playwright(url):
    """Extract text using Playwright for JS-heavy sites"""
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until='networkidle')
            
            # Wait for potential AJAX calls
            page.wait_for_timeout(3000)
            
            # Try clicking common fee-related elements
            fee_selectors = [
                'text="fees"', 'text="fee structure"', 'text="admission"',
                '[href*="fee"]', '[href*="admission"]', '.fee', '#fees'
            ]
            
            for selector in fee_selectors:
                try:
                    if page.locator(selector).count() > 0:
                        page.locator(selector).first.click()
                        page.wait_for_timeout(2000)
                        break
                except:
                    continue
            
            text = page.inner_text('body')
            browser.close()
            return text
    except Exception as e:
        print(f"Playwright extraction failed: {e}")
        return None

def extract_all_text(url):
    """Extract all text from URL with fallback strategies"""
    
    try:
        # First try: Standard requests
        response = requests.get(url, verify=False, timeout=30)
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            # PDF extraction
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                f.write(response.content)
                temp_path = f.name
            
            text = ""
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            os.unlink(temp_path)
            return text, "pdf"
            
        else:
            # HTML extraction
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            
            # Check if content seems JS-heavy (low text content)
            if len(text.strip()) < 500 or 'javascript' in text.lower():
                print("Detected JS-heavy site, trying Playwright...")
                playwright_text = extract_with_playwright(url)
                if playwright_text and len(playwright_text) > len(text):
                    return playwright_text, "playwright"
            
            return text, "requests"
            
    except Exception as e:
        # Fallback to Playwright
        print(f"Standard extraction failed: {e}, trying Playwright...")
        playwright_text = extract_with_playwright(url)
        if playwright_text:
            return playwright_text, "playwright_fallback"
        return f"Extraction error: {e}", "error"



def process_chunk_with_prompt(chunk, prompt_template, college_name):
    """Process single chunk with given prompt"""
    try:
        prompt = PromptTemplate(
            input_variables=["college_name", "fee_data"],
            template=prompt_template
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"college_name": college_name, "fee_data": chunk})
        return result.strip()
    except Exception as e:
        return f"Error: {e}"

def extract_all_fees(fee_chunks, college_name):
    """Extract all fee-related data in tabular format"""
    
    prompt_template = """Extract ALL fee-related information from {college_name} and format as a clean table:
{fee_data}

Return ONLY fees that are actually present in the format of table with all related parameters like fee type, amount, duration, etc.
Include only fees mentioned in the data. Skip any fees not present. Use exact amounts from the source.
"""
    
    print("Extracting all fee data...")
    all_fees = []
    for i, chunk in enumerate(fee_chunks[:15]):
        print(f"Processing chunk {i+1}/{min(15, len(fee_chunks))}")
        result = process_chunk_with_prompt(chunk, prompt_template, college_name)
        if result and not result.startswith("Error") and result.strip() and "|" in result:
            all_fees.append(result)
    
    return "\n\n".join(all_fees)

def create_vectorstore_and_search_enhanced(text, custom_query=None):
    """Create vectorstore and use enhanced semantic search"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from text")
    
    # Create vectorstore from all chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Use custom query or multiple targeted queries
    queries = [
        custom_query or "fees tuition cost charges admission program course semester annual amount rupees",
        "admission fee structure tuition charges",
        "course fees program fees semester charges",
        "any fee related data along with fee related parameters like caution money deposit mess hostel",
        "development fee caution money hostel charges"
    ]
    
    all_relevant_chunks = []
    for query in queries:
        fee_docs = vectorstore.similarity_search(query, k=8)
        for doc in fee_docs:
            if doc.page_content not in all_relevant_chunks:
                all_relevant_chunks.append(doc.page_content)
    
    print(f"Retrieved {len(all_relevant_chunks)} unique fee-related chunks using enhanced search")
    return all_relevant_chunks[:20]  # Limit to top 20 chunks

def langchain_fee_extractor(url, college_name, custom_query=None):
    """Enhanced LangChain-based semantic fee extractor with agent and Playwright fallback"""
    
    print(f"Step 1: Extracting all text from {url}")
    extraction_result = extract_all_text(url)
    
    if isinstance(extraction_result, tuple):
        raw_text, extraction_method = extraction_result
    else:
        raw_text, extraction_method = extraction_result, "unknown"
    
    if raw_text.startswith("Extraction error"):
        return {"error": raw_text}
    
    print(f"Step 2: Creating vectorstore and using enhanced semantic search...")
    fee_chunks = create_vectorstore_and_search_enhanced(raw_text, custom_query)
    
    if not fee_chunks:
        return {"error": "No fee chunks found"}
    
    print(f"Step 3: Extracting all fee data using LLM...")
    all_fees_data = extract_all_fees(fee_chunks, college_name)
    
    return {
        "college": college_name,
        "url": url,
        "total_text_length": len(raw_text),
        "extraction_method": f"langchain_enhanced_{extraction_method}",
        "all_fees_data": all_fees_data,
        "chunks_processed": len(fee_chunks)
    }

def save_to_csv(result):
    """Save extraction result to CSV"""
    if "error" in result:
        df = pd.DataFrame([{
            "college": result.get("college", "Unknown"),
            "url": result.get("url", ""),
            "error": result["error"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
    else:
        df = pd.DataFrame([{
            "college": result["college"],
            "url": result["url"],
            "all_fees_data": result["all_fees_data"],
            "text_length": result["total_text_length"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
    
    filename = f"fee_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    return filename

if __name__ == "__main__":
    # Example with custom query
    custom_query = "Find detailed fee structure including admission fees, tuition fees, and any semester-wise or annual charges orr any fee related data alon with fee related parameters for UCEED"
    
    result = langchain_fee_extractor(
        "https://home.iitd.ac.in/uploads/ug/24-25/Fee%20Structure24-25.pdf", 
        "iit delhi",
        custom_query=custom_query
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_to_csv(result)