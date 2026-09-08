import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from environment variables."
        )

    return api_key


def get_model_name():
    return os.getenv("LUNA_MODEL", "gpt-5.6-luna")


def get_chat_model(temperature=0.1):
    return ChatOpenAI(
        api_key=get_api_key(),
        model=get_model_name(),
        temperature=temperature,
        max_tokens=700,
        timeout=45,
        max_retries=2,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        api_key=get_api_key(),
        model="text-embedding-3-small",
    )