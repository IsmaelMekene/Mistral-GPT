from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument

# Define a separator for better visual separation
separator = "#" * 150

# Print a success message to indicate that the imports were successful
print(f"{separator}\nImport Successful\n{separator}")

# Define the path to the PDF document you want to process
path_doc = [
    "/Users/me_teor21/Workspace/Mistral-GPT/data/doc.pdf"  # noqa: E501
]

# Create a Weaviate document store with specified host,
# port, and embedding dimension
document_store = WeaviateDocumentStore(
    host="http://localhost", port=8080, embedding_dim=384
)  # noqa: E501

# Print information about the Weaviate document store
print(f"{separator}\nDocument Store: {document_store}\n{separator}")

# Create a PyPDFToDocument converter to extract text from the PDF document
converter = PyPDFToDocument()
# Print information about the converter
print(f"{separator}\nConverter: {converter}\n{separator}")

# Run the converter to extract text from the specified PDF document
output = converter.run(paths=path_doc)
docs = output["documents"]

# Process the extracted documents to create
# a structured format for further analysis
final_doc = []
for doc in docs:
    # print(f"{separator}\n{doc.text}\n{separator}")
    new_doc = {"content": doc.text, "meta": doc.metadata}
    final_doc.append(new_doc)

# Create a preprocessor with specific
# configurations to clean and split the text
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
)
# Print information about the preprocessor
print(f"{separator}\nPreprocessor: {preprocessor}\n{separator}")

# Process the final documents using the preprocessor
preprocessed_docs = preprocessor.process(final_doc)

# Write the preprocessed documents to the Weaviate document store
document_store.write_documents(preprocessed_docs)

# Create an EmbeddingRetriever with a specified embedding model
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    # embedding_model="sentence-transformers/multilingual-e5-base"
)

# Print information about the retriever
print(f"{separator}\nRetriever: {retriever}\n{separator}")

# Update the embeddings in the Weaviate document store using the retriever
document_store.update_embeddings(retriever)

# Print a message indicating that the embeddings have been updated
print(f"{separator}\nEmbeddings Updated\n{separator}")
