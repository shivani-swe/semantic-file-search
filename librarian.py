import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import docx2txt

load_dotenv()

local_ef= embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client= chromadb.PersistentClient(path='./db')
collection= client.get_or_create_collection(name="my_library", embedding_function=local_ef)

llm= ChatGroq(model='llama-3.3-70b-versatile', api_key=os.getenv("GROQ_API_KEY"))

def add_to_lib(text, doc_id):
    chunks= [c.strip() for c in text.split('\n\n') if len(c.strip()) >10]
    ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
    metadata= [{"source": doc_id} for _ in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadata)

def search_lib(query):
    results= collection.query(query_texts=[query], n_results=3)
    return results['documents'][0]

st.set_page_config(page_title="The Digital Librarian", layout="wide")
st.title("The Digital Librarian (Local RAG)")

with st.sidebar:
    st.header("Manage library")
    if st.button("Clear Library"):
        client.delete_collection("my_library")
        collection= client.create_collection(name="my_library", embedding_function=local_ef)
        st.warning("Library wiped clean! You can now upload fresh files.")
    st.divider()

    
    st.header("Insert Document")
    uploaded_files= st.file_uploader("Upload text files", type=['txt', 'md', 'docx', 'doc'], accept_multiple_files=True)

    if st.button("Index Files"):
        for file in uploaded_files:
            file_extension= file.name.split('.')[-1].lower()

            if file_extension=="docx":
                content= docx2txt.process(file)
            else:
                raw_bytes= file.read()
                try:
                    content=raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    st.toast(f"Note: {file.name} isn't UTF-8. Using fallback.")
                    content= raw_bytes.decode("latin-1")
                    
            add_to_lib(content, file.name)
        st.success(f"Indexed {len(uploaded_files)} files locally.")

st.header("Search the document")
query= st.text_input("Ask a question about your document:")

if query:
    with st.spinner("Searching local vault"):
        context_chunks= search_lib(query)
        context_text="\n\n".join(context_chunks)

        prompt= f"""You are a helpful librarian. Use the provided context from the user's private files 
        to answer the question. If the answer isn't in the context, say you don't know.
        
        CONTEXT:
        {context_text}

        QUESTION:
        {query}
        """

        response= llm.invoke(prompt)

        st.markdown("Librarian's Answer: ")
        st.write(response.content)

        with st.expander("See Source chunks"):
            for i, chunk in enumerate(context_chunks):
                st.info(f"Chunk {i+1}: {chunk}")