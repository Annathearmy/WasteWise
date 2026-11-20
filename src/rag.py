import os
from pathlib import Path
import chromadb
import nest_asyncio

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.response_synthesizers import get_response_synthesizer

from src.shared_embedder import SharedLlamaIndexEmbedding

nest_asyncio.apply()

DEFAULT_PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
PROJECT_PATH = os.environ.get("WASTEWISE_PROJECT_PATH", DEFAULT_PROJECT_PATH)

CHROMA_PATH = os.path.join(PROJECT_PATH, "Database", "chromadb")

_retriever = None
_response_synthesizer = None


def _ensure_rag_initialized():
    global _retriever, _response_synthesizer
    if _retriever is not None and _response_synthesizer is not None:
        return

    # Connect to the existing ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    knowledge_collection = chroma_client.get_collection(
        name="wastewise_knowledge"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=knowledge_collection
    )

    # Reuse the single shared ONNX embedding model
    embed_model = SharedLlamaIndexEmbedding()

    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model
    )

    _retriever = index.as_retriever(similarity_top_k=8)

    # Connect Gemini
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except Exception:
            pass

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your .env file or environment.")

    llm = GoogleGenAI(
        model="gemini-3.5-flash",
        api_key=api_key
    )

    Settings.llm = llm

    # Configure the grounding prompt
    qa_prompt = PromptTemplate(
        """
        You are WasteWise, a waste management information assistant.

        Answer using ONLY the provided context.

        Rules:
1. Do not use outside knowledge or make assumptions.
2. Do not invent rules, responsibilities, dates or legal requirements.
3. If the context is insufficient, clearly say so.
4. Do not present older rules as the latest legal requirements.
5. Keep the answer clear and structured.
6. Use simple language that ordinary users can understand.

        Context:
        {context_str}

        Question:
        {query_str}

        Answer:
        """
    )

    _response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=qa_prompt,
        response_mode="compact"
    )
    print("WasteWise RAG module initialized (shared ONNX runtime)")


def ask_wastewise(question):
    _ensure_rag_initialized()

    expanded_question = (
        question
        + " Functions and responsibilities under Indian waste management rules. "
        + "Consumer, producer, collection, segregation, refurbishment, recycling "
        + "and environmentally friendly disposal."
    )

    original_results = _retriever.retrieve(question)
    expanded_results = _retriever.retrieve(expanded_question)

    combined_results = []
    seen_ids = set()

    for result in original_results + expanded_results:
        node_id = result.node.node_id

        if node_id not in seen_ids:
            seen_ids.add(node_id)
            combined_results.append(result)

    ranked_results = sorted(
        combined_results,
        key=lambda result: result.score or 0,
        reverse=True
    )

    response = _response_synthesizer.synthesize(
        query=question,
        nodes=ranked_results[:8]
    )

    sources = []
    seen_sources = set()

    for source in response.source_nodes:
        metadata = source.node.metadata
        file_name = metadata.get("file_name", "Unknown source")
        page_number = metadata.get("page_number", "Unknown page")

        source_key = (file_name, page_number)

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "file_name": file_name,
                "page_number": page_number
            })

    return {
        "answer": response.response,
        "sources": sources
    }