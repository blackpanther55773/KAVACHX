import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import get_embeddings
DOCUMENTS = [
    Document(
        page_content=(
            "The Government of India's Emergency Response Support System (ERSS) "
            "provides a nationwide emergency number, 112, for citizens in distress. "
            "ERSS can handle emergencies involving police, health, fire, women "
            "protection and other emergency services. During immediate danger, "
            "move toward a safer location when possible and seek immediate "
            "emergency assistance."
        ),
        metadata={
            "title": "Ministry of Home Affairs - Emergency Response Support System (ERSS)",
            "url": "https://www.mha.gov.in/en/commoncontent/emergency-response-support-system-erss",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Government of India has implemented ERSS 112 as a pan-India "
            "emergency response system. It supports emergency assistance through "
            "multiple channels and connects actionable incidents with relevant "
            "emergency response services. In an immediate safety emergency, "
            "contact emergency services rather than relying on an AI assistant."
        ),
        metadata={
            "title": "MHA - ERSS 112",
            "url": "https://www.mha.gov.in/en/divisionofmha/women-safety-division/emergency-response-support-system-erss",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Ministry of Health and Family Welfare, Government of India, "
            "publishes Guidelines for Medico-Legal Care for Survivors/Victims of "
            "Sexual Violence. The guidelines address health consequences, the role "
            "of health professionals, medical examination and reporting, "
            "psycho-social care, and interaction with other agencies."
        ),
        metadata={
            "title": "MoHFW - Guidelines for Medico-Legal Care for Survivors/Victims of Sexual Violence",
            "url": "https://www.mohfw.gov.in/sites/default/files/953522324.pdf",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Government of India's National Cyber Crime Reporting Portal "
            "provides a mechanism for citizens to report cyber crime. For cyber "
            "financial fraud, the portal states that immediate reporting can be "
            "made through the national cyber crime helpline number 1930. Users "
            "should avoid sharing passwords, OTPs, PINs or banking credentials."
        ),
        metadata={
            "title": "National Cyber Crime Reporting Portal - Government of India",
            "url": "https://www.cybercrime.gov.in/",
            "domain": "Cyber Fraud",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The National Cyber Crime Reporting Portal is an initiative of the "
            "Government of India for reporting cyber crime complaints online. "
            "For financial cyber fraud, the portal provides the national cyber "
            "crime helpline number 1930 for immediate reporting."
        ),
        metadata={
            "title": "I4C - Cyber Crime Reporting",
            "url": "https://www.cybercrime.gov.in/Accept.aspx",
            "domain": "Cyber Fraud",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Government of India's Emergency Response Support System (ERSS) "
            "provides a nationwide emergency response number 112. The system "
            "coordinates emergency assistance for services including health, "
            "police and fire. For a serious or potentially life-threatening "
            "medical emergency, seek professional emergency assistance immediately "
            "and follow instructions from qualified emergency personnel."
        ),
        metadata={
            "title": "Ministry of Home Affairs - ERSS 112",
            "url": "https://www.mha.gov.in/en/commoncontent/emergency-response-support-system-erss",
            "domain": "Medical Emergency",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Government of India National Portal provides an official helpline "
            "directory. It lists 112 as the integrated Emergency Response Support "
            "System for police, fire, rescue and health services and lists 1930 "
            "as the cyber crime helpline."
        ),
        metadata={
            "title": "National Portal of India - Helpline Directory",
            "url": "https://www.india.gov.in/directory/helpline",
            "domain": "Medical Emergency",
            "country": "India",
            "authority": "Government of India",
        },
    ),
]

@st.cache_resource(show_spinner=False)
def get_vector_store():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(DOCUMENTS)

    return FAISS.from_documents(
        chunks,
        get_embeddings(),
    )


def retrieve_context(query, domain, k=4):
    store = get_vector_store()

    documents = store.similarity_search(
        query,
        k=k,
    )

    filtered = [
        document
        for document in documents
        if document.metadata.get("domain") == domain
    ]

    documents = filtered or documents

    context = []
    sources = []
    seen = set()

    for document in documents[:k]:
        metadata = document.metadata

        context.append(
            f"Source: {metadata['title']}\n"
            f"URL: {metadata['url']}\n"
            f"Guidance: {document.page_content}"
        )

        if metadata["url"] not in seen:
            sources.append({
                "title": metadata["title"],
                "url": metadata["url"],
            })
            seen.add(metadata["url"])

    return "\n\n".join(context), sources