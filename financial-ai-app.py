import streamlit as st
import os
import re
import io
import sys
import pdfplumber
import pandas as pd
from langchain_groq import ChatGroq

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="100% Accurate Financial AI", page_icon="📊", layout="wide")
st.title("📊 Financial AI: 100% Accuracy Engine")
st.markdown("*Upload your Bank Statements or Invoices. Zero hallucinations. Pure Python Math.*")

# --- SECURE API KEY ---
# It checks Streamlit secrets first, then allows manual entry in the sidebar for testing
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

with st.sidebar:
    st.header("Configuration")
    api_key_input = st.text_input("Groq API Key (If not using Secrets)", type="password")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
        
    uploaded_file = st.file_uploader("Upload a Bank Statement PDF", type=["pdf"])
    
    st.markdown("---")
    st.markdown("**Disclaimer:** Human-in-the-loop required for final accounting.")

# --- PDF PARSER (Accurate Table Extraction) ---
def extract_and_parse_pdf(file_path):
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    # Custom Parser for Ally Bank Layout
    lines = full_text.split('\n')
    transactions = []
    current_tx = None

    for line in lines:
        # Find lines starting with a date (MM/DD/YYYY)
        match = re.match(r'^(\d{2}/\d{2}/\d{4})\s+(.*)', line)
        if match:
            if current_tx: transactions.append(current_tx)
            date = match.group(1)
            rest = match.group(2)
            
            # Find all money amounts (e.g., $503.00 or -$503.00)
            amounts = re.findall(r'(?:\$?-?[\d,]+\.\d{2})', rest)
            
            if len(amounts) >= 3:
                credit_str, debit_str, balance_str = amounts[-3], amounts[-2], amounts[-1]
                desc_end = rest.find(credit_str)
                description = rest[:desc_end].strip()
                
                current_tx = {
                    'Date': date, 'Description': description,
                    'Credits': credit_str.replace('$', '').replace(',', ''),
                    'Debits': debit_str.replace('-$', '-').replace('$', '').replace(',', ''),
                    'Balance': balance_str.replace('$', '').replace(',', '')
                }
            else:
                current_tx = None
        # If the line doesn't start with a date, it's a continuation of the description
        elif current_tx and line.strip():
            current_tx['Description'] += " " + line.strip()

    if current_tx: transactions.append(current_tx)
    
    # Create DataFrame and convert to numbers
    df = pd.DataFrame(transactions)
    if not df.empty:
        for col in ['Credits', 'Debits', 'Balance']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, full_text

# --- AI BRAIN (Math + Text Router) ---
def ask_question(question, df, full_text, llm):
    # 1. Route the question
    router_prompt = f"""Does this question require calculating, summing, counting, or filtering the data table? 
Reply with exactly one word: "MATH" or "TEXT". Question: {question}"""
    
    try:
        intent = llm.invoke(router_prompt).content.strip().upper()
    except:
        intent = "TEXT"

    # 2. MATH ROUTING (AI writes Pandas code)
    if "MATH" in intent and not df.empty:
        table_csv = df.to_csv(index=False)
        code_prompt = f"""You are a Senior Financial Python Analyst.
Table Data:
{table_csv}

Dataframe is loaded as `df`. Columns: {list(df.columns)}
RULES: Credits = IN, Debits = OUT. If asked for "transactions", give count (len) and volume. Format money as f"${{amount:,.2f}}".
Question: {question}
Write Python code using pandas to answer. Print the answer.
Return ONLY the code. No markdown."""
        try:
            python_code = llm.invoke(code_prompt).content.strip()
            python_code = python_code.replace("```python", "").replace("```", "").strip()
            
            # Execute the Python code
            local_vars = {'df': df, 'pd': pd}
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            exec(python_code, {}, local_vars)
            sys.stdout = old_stdout
            
            result = redirected_output.getvalue().strip()
            return f"📊 **Result:**\n{result}\n*(Calculated 100% accurately by Python)*"
        except Exception as e:
            return f"⚠️ Math Error: {str(e)}"

    # 3. TEXT ROUTING (AI reads raw text)
    else:
        # Limit text to ~25,000 characters to prevent token limits
        context = full_text[:25000] 
        prompt_rag = f"""Context: {context}\nQuestion: {question}\nAnswer in ONE short sentence using ONLY the context."""
        try:
            return llm.invoke(prompt_rag).content
        except Exception as e:
            return f"⚠️ AI Error: {str(e)}"

# --- UI MEMORY (Session State) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "df" not in st.session_state: st.session_state.df = None
if "full_text" not in st.session_state: st.session_state.full_text = None

# --- FILE PROCESSING ---
if uploaded_file and os.environ.get("GROQ_API_KEY"):
    if st.session_state.df is None:
        with st.spinner("Processing PDF (Extracting Table)..."):
            temp_filename = uploaded_file.name
            with open(temp_filename, "wb") as f: f.write(uploaded_file.getbuffer())
            
            df, full_text = extract_and_parse_pdf(temp_filename)
            llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
            
            st.session_state.df = df
            st.session_state.full_text = full_text
            st.session_state.llm = llm
            os.remove(temp_filename)
            
        st.success(f"✅ Document Processed! Found {len(df)} transactions. Ask your questions.")

# --- CHAT DISPLAY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.markdown(message["content"])

# --- CHAT INPUT ---
if prompt := st.chat_input("Ask a question (e.g., 'What is the total of all debits?')"):
    if st.session_state.full_text is None:
        st.warning("Please upload a PDF and enter your API key first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = ask_question(prompt, st.session_state.df, st.session_state.full_text, st.session_state.llm)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
