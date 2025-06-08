import openai
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Set API base for OpenRouter
openai.api_base = "https://openrouter.ai/api/v1"

# Retrieve API key securely
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_KEY = "**key**"  # Replace with your actual key
# "  # Replace with your actual key

openai.api_key = OPENROUTER_API_KEY

# Step 1: Load PDFs into LangChain
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)

# Step 2: Split documents into chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)

# Step 3: Embed Text Chunks & Store in FAISS
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)

# Step 4: Set Up Retriever from FAISS
retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant chunks

# Step 5: Function to Ask Questions, Using **Both AI Model + Knowledge Base**
def ask_question():
    user_question = input("\nEnter your question: ")

    # Retrieve knowledge base documents relevant to the query
    relevant_docs = retriever.invoke(user_question)
    context = "\n".join([doc.page_content for doc in relevant_docs])  # Combine text

    messages = [
        {"role": "system", "content": "You are an AI assistant powered by DeepSeek with additional knowledge from a document database."},
        {"role": "user", "content": f"Context from retrieved documents:\n{context}\n\nUser Query: {user_question}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=messages,
            max_tokens=1500,  # Avoid excessive token usage
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://yourwebsite.com",
                "X-Title": "DeepSeek AI Knowledge Assistant"
            }
        )

        # Extract and print the final optimized response
        reply = response["choices"][0]["message"]["content"]
        print("\n🧠 Optimized AI Response (LLM + Knowledge Base):", reply)

    except openai.error.RateLimitError:
        print("⚠️ Rate limit exceeded! Please wait before making another request.")
    except openai.error.APIError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"🚨 Unexpected Error: {e}")

# Continuous interactive query loop
while True:
    ask_question()
    continue_prompt = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
    if continue_prompt != "yes":
        print("Goodbye! 👋")
        break
