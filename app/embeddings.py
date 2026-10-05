from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=s.embedding_model)
    if s.embedding_provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model=s.embedding_model)
    if s.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=f"models/{s.embedding_model}", output_dimensionality=s.embedding_dim)
    if s.embedding_provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=s.embedding_model)
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=s.embedding_dim)
