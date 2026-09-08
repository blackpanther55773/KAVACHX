import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm_config import get_embeddings


DOCUMENTS = [
    Document(
        page_content=(
            "The Ministry of Home Affairs, Government of India, has a dedicated "
            "Women Safety Division dealing with matters related to safety and "
            "security of women, crimes against women, and measures for prevention "
            "and response. Women facing immediate danger should move toward a "
            "safer location and seek help from appropriate emergency or "
            "law-enforcement services."
        ),
        metadata={
            "title": "MHA - Women Safety Division",
            "url": "https://www.mha.gov.in/en/divisionofmha/women-safety-division",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Ministry of Home Affairs provides information and initiatives "
            "related to the safety and security of women in India. These "
            "initiatives address prevention of crimes against women, response "
            "mechanisms, and strengthening institutional support for women."
        ),
        metadata={
            "title": "MHA - Safety and Security of Women",
            "url": "https://www.mha.gov.in/en/divisionofmha/women-safety-division",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "The Ministry of Health and Family Welfare, Government of India, "
            "publishes Guidelines for Medico-Legal Care for Survivors/Victims "
            "of Sexual Violence. The guidelines address health consequences, "
            "the role of health professionals, medical examination and reporting, "
            "psycho-social care, and interaction with other agencies."
        ),
        metadata={
            "title": (
                "MoHFW - Guidelines for Medico-Legal Care "
                "for Survivors/Victims of Sexual Violence"
            ),
            "url": "https://www.mohfw.gov.in/sites/default/files/953522324.pdf",
            "domain": "Women Safety",
            "country": "India",
            "authority": "Government of India",
        },
    ),
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
            "title": "MHA - ERSS 112 Supporting Emergency Response",
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
            "The Government of India's National Cyber Crime Reporting Portal "
            "provides a mechanism for citizens to report cyber crime. For cyber "
            "financial fraud, the portal provides the national cyber crime "
            "helpline number 1930 for immediate reporting. Users should avoid "
            "sharing passwords, OTPs, PINs or banking credentials."
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
            "For suspected cyber financial fraud, citizens should report the "
            "incident promptly through the National Cyber Crime Reporting Portal "
            "and use the cyber crime helpline 1930. Users should not share OTPs, "
            "passwords, PINs, card details or other banking credentials with "
            "unknown persons."
        ),
        metadata={
            "title": "I4C - Financial Cyber Fraud Reporting",
            "url": "https://www.cybercrime.gov.in/",
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
            "title": "MHA - ERSS 112 Medical Emergency Support",
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
            "System for police, fire, rescue and health services. In a serious "
            "medical emergency, seek professional medical assistance immediately."
        ),
        metadata={
            "title": "National Portal of India - Helpline Directory",
            "url": "https://www.india.gov.in/directory/helpline",
            "domain": "Medical Emergency",
            "country": "India",
            "authority": "Government of India",
        },
    ),
    Document(
        page_content=(
            "In a serious or potentially life-threatening medical situation, "
            "professional emergency assistance should be sought immediately. "
            "An AI assistant should not replace qualified medical professionals "
            "or emergency responders. Follow instructions provided by trained "
            "emergency personnel."
        ),
        metadata={
            "title": "Government of India - Emergency Medical Guidance",
            "url": "https://www.india.gov.in/",
            "domain": "Medical Emergency",
            "country": "India",
            "authority": "Government of India",
        },
    ),
]


DOMAIN_PRIORITY = {
    "Women Safety": [
        "women safety",
        "safety and security of women",
        "crime against women",
        "women help desk",
        "sexual violence",
        "medico-legal",
        "erss",
    ],
    "Cyber Fraud": [
        "national cyber crime reporting portal",
        "i4c",
        "financial cyber fraud",
        "cyber crime",
    ],
    "Medical Emergency": [
        "medical emergency",
        "medical",
        "health",
        "emergency response",
        "erss",
        "national portal",
    ],
}


def get_priority_score(document, domain):
    title = document.metadata.get("title", "").lower()
    keywords = DOMAIN_PRIORITY.get(domain, [])

    for index, keyword in enumerate(keywords):
        if keyword in title:
            return len(keywords) - index

    return 0


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
        k=10,
    )

    filtered = [
        document
        for document in documents
        if document.metadata.get("domain") == domain
    ]

    if filtered:
        documents = filtered

    documents.sort(
        key=lambda document: get_priority_score(
            document,
            domain,
        ),
        reverse=True,
    )

    documents = documents[:k]

    context = []
    sources = []
    seen = set()

    for document in documents:
        metadata = document.metadata

        context.append(
            f"Source: {metadata.get('title', 'Unknown source')}\n"
            f"URL: {metadata.get('url', '')}\n"
            f"Guidance: {document.page_content}"
        )

        url = metadata.get("url", "")

        if url and url not in seen:
            sources.append({
                "title": metadata.get(
                    "title",
                    "Unknown source",
                ),
                "url": url,
            })
            seen.add(url)

    return "\n\n".join(context), sources