from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from utlies import pdf_to_text
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
# def generate_tokens(doc_path):#
#     cleaned_text = pdf_to_text(
#         pdf_path=doc_path)
#     text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=15)
#     chunks = text_splitter.split_text(cleaned_text)
#     return chunks
def perform_semantic_chunking(text):
    # Initialize the embedding model (using HuggingFace embeddings here)
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Initialize the SemanticChunker
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")

    # Create chunks based on the semantic content
    semantic_chunks = semantic_chunker.create_documents([text])

    return semantic_chunks

text=pdf_to_text(pdf_path=r"C:\Users\janag\Downloads\Pdf2.pdf")
all_chunks=perform_semantic_chunking(text)
all_text=[]
for x in all_chunks:
    all_text.append(x.page_content)
print(all_text)
embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')
vector_store = FAISS.from_texts(all_text, embeddings)
