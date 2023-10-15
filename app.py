import re

import uvicorn
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from haystack import Pipeline
from haystack.document_stores import WeaviateDocumentStore
from haystack.nodes import (  # noqa: E501
    AnswerParser,
    EmbeddingRetriever,
    PromptNode,
    PromptTemplate,
)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


# Use FastAPI's dependency injection for creating document store and query pipeline # noqa: E501
def get_document_store():
    return WeaviateDocumentStore(
        host="http://localhost", port=8080, embedding_dim=384
    )  # noqa: E501


def get_query_pipeline():
    document_store = get_document_store()

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])  # noqa: E501
    query_pipeline.add_node(
        component=PromptNode(
            model_name_or_path="model/mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            max_length=1000,
            default_prompt_template=PromptTemplate(
                prompt="Your prompt here", output_parser=AnswerParser()
            ),
        ),
        name="PromptNode",
        inputs=["Retriever"],
    )

    return query_pipeline


# This endpoint can be used to perform the actual question answering
@app.post("/get_answer", response_class=JSONResponse)
async def get_answer(
    question: str = Form(...),
    document_store=Depends(get_document_store),
    query_pipeline=Depends(get_query_pipeline),
):
    json_response = query_pipeline.run(
        query=question, params={"Retriever": {"top_k": 5}}
    )  # noqa: E501
    answers = json_response["answers"]

    # Extract the answer
    for ans in answers:
        answer = ans.answer
        break

    # Extract relevant documents and their content
    documents = json_response["documents"]
    document_info = [document.content for document in documents]

    # Split the answer into sentences
    sentences = re.split(r"(?<=[.!?])\s", answer)

    # Filter out incomplete sentences
    complete_sentences = [
        sentence for sentence in sentences if re.search(r"[.!?]$", sentence)
    ]  # noqa: E501

    # Rejoin the complete sentences into a single string
    updated_answer = " ".join(complete_sentences)

    # Prepare a string with relevant documents' content
    relevant_documents = "\n".join(
        [
            f"Document {i + 1} Content:\n{content}"
            for i, content in enumerate(document_info)  # noqa: E501
        ]
    )

    return {"answer": updated_answer, "relevant_documents": relevant_documents}


# Define an endpoint for the homepage
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
