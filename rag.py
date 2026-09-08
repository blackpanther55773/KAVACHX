import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import get_embeddings


DOCUMENTS = [
    Document(
        page_content=(
            "The Ministry of Home Affairs, Government of India, has a dedicated "
            "Women Safety Division dealing with safety and security of women, "
            "crimes against women, prevention and response. Women facing immediate "
            "danger should move to a safer location and seek appropriate emergency "
            "or law-enforcement assistance."
        ),
        metadata={
            "title": "MHA - Women Safety Division",
            "url": "https://www.mha.gov.in/en/divisionofmha/women-safety-division",
            "domain": "Women Safety",
            "priority": 100,
        },
    ),

    Document(
        page_content=(
            "The Ministry of Home Affairs provides information and initiatives "
            "related to the safety and security of women in India. These initiatives "
            "address prevention of crimes against women and institutional support."
        ),
        metadata={
            "title": "MHA - Crime Against Women / Women Safety",
            "url": "https://www.mha.gov.in/en/commoncontent/crime-against-women-0",
            "domain": "Women Safety",
            "priority": 90,
        },
    ),

    Document(
        page_content=(
            "The Ministry of Health and Family Welfare publishes Guidelines for "
            "Medico-Legal Care for Survivors/Victims of Sexual Violence. The "
            "guidelines cover health consequences, medical examination, reporting, "
            "psycho-social care and interaction with other agencies."
        ),
        metadata={
            "title": "MoHFW - Guidelines for Medico-Legal Care",
            "url": "https://www.mohfw.gov.in/sites/default/files/953522324.pdf",
            "domain": "Women Safety",
            "priority": 80,
        },
    ),

    Document(
        page_content=(
            "The Government of India's Emergency Response Support System provides "
            "the nationwide emergency number 112. ERSS supports emergencies "
            "involving police, health, fire, women protection and other emergency "
            "services. During immediate danger, seek emergency assistance."
        ),
        metadata={
            "title": "MHA - ERSS 112 Supporting Emergency Response",
            "url": "https://www.mha.gov.in/en/commoncontent/emergency-response-support-system-erss",
            "domain": "Women Safety",
            "priority": 40,
        },
    ),

    Document(
        page_content=(
            "The Government of India's National Cyber Crime Reporting Portal "
            "provides a mechanism for citizens to report cyber crime. For financial "
            "cyber fraud, citizens can use the national cyber crime helpline 1930."
        ),
        metadata={
            "title": "National Cyber Crime Reporting Portal",
            "url": "https://www.cybercrime.gov.in/",
            "domain": "Cyber Fraud",
            "priority": 100,
        },
    ),

    Document(
        page_content=(
            "The National Cyber Crime Reporting Portal is an initiative of the "
            "Government of India for reporting cyber crime complaints online. "
            "Financial cyber fraud should be reported promptly using the portal "
            "and helpline 1930."
        ),
        metadata={
            "title": "I4C - Cyber Crime Reporting",
            "url": "https://www.cybercrime.gov.in/Accept.aspx",
            "domain": "Cyber Fraud",
            "priority": 90,
        },
    ),

    Document(
        page_content=(
            "For suspected cyber financial fraud, citizens should report the "
            "incident promptly through the National Cyber Crime Reporting Portal "
            "and use the cyber crime helpline 1930. Never share OTPs, passwords, "
            "PINs or banking credentials with unknown persons."
        ),
        metadata={
            "title": "I4C - Financial Cyber Fraud Reporting",
            "url": "https://www.cybercrime.gov.in/",
            "domain": "Cyber Fraud",
            "priority": 80,
        },
    ),

    Document(
        page_content=(
            "The Government of India's Emergency Response Support System provides "
            "the nationwide emergency response number 112. It coordinates emergency "
            "assistance including health, police and fire services. For a serious "
            "medical emergency, seek professional emergency assistance immediately."
        ),
        metadata={
            "title": "MHA - ERSS 112 Medical Emergency Support",
            "url": "https://www.mha.gov.in/en/commoncontent/emergency-response-support-system-erss",
            "domain": "Medical Emergency",
            "priority": 100,
        },
    ),

    Document(
        page_content=(
            "The Government of India National Portal provides an official helpline "
            "directory and lists 112 as the integrated Emergency Response Support "
            "System for police, fire, rescue and health services."
        ),
        metadata={
            "title": "National Portal of India - Helpline Directory",
            "url": "https://www.india.gov.in/directory/helpline",
            "domain": "Medical Emergency",
            "priority": 90,
        },
    ),

    Document(
        page_content=(
            "In a serious or potentially life-threatening medical situation, "
            "professional emergency assistance should be sought immediately. "
            "An AI assistant should not replace qualified medical professionals "
            "or emergency responders."
        ),
        metadata={
            "title": "Government of India - Emergency Medical Guidance",
            "url": "https://www.india.gov.in/",
            "domain": "Medical Emergency",
            "priority": 80,
        },
    ),
]


@st.cache_resource(show_spinner=False)
def get_vector_store(domain):
    domain_documents = [
        document
        for document in DOCUMENTS
        if document.metadata.get("domain") == domain
    ]

    if not domain_documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=80,
    )

    chunks = splitter.split_documents(domain_documents)

    return FAISS.from_documents(
        chunks,
        get_embeddings(),
    )


def retrieve_context(query, domain, k=4):
    domain_documents = [
        document
        for document in DOCUMENTS
        if document.metadata.get("domain") == domain
    ]

    if not domain_documents:
        return "", []

    store = get_vector_store(domain)

    results = store.similarity_search_with_score(
        query,
        k=min(k, len(domain_documents)),
    )

    results.sort(
        key=lambda item: (
            -item[0].metadata.get("priority", 0),
            item[1],
        )
    )

    context = []
    sources = []
    seen_urls = set()

    for document, score in results[:k]:
        metadata = document.metadata

        context.append(
            f"Source: {metadata.get('title', 'Unknown source')}\n"
            f"URL: {metadata.get('url', '')}\n"
            f"Guidance: {document.page_content}"
        )

        url = metadata.get("url", "")

        if url and url not in seen_urls:
            sources.append(
                {
                    "title": metadata.get(
                        "title",
                        "Unknown source",
                    ),
                    "url": url,
                }
            )
            seen_urls.add(url)

    return "\n\n".join(context), sources