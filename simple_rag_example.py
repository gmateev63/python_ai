from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Step 1: Load your document
loader = TextLoader("company_policy.txt")
documents = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3: Create embeddings and store in vector DB
embedding_model = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embedding_model)

# Step 4: Set up retrieval + LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever()
)

# Step 5: Ask a question
answer = qa_chain.run("What is our refund policy for digital products?")
print(answer)