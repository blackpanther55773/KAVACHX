# app.py
import streamlit as st
from workflow import run_workflow

st.set_page_config(
    page_title="KAVACHX",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
body { background: #f7f9fc; }
.header {
    background: linear-gradient(135deg,#14213d,#243b64);
    color: white;
    padding: 1.3rem 1.5rem;
    border-radius: 18px;
    margin-bottom: 1rem;
}
.notice {
    padding: .8rem 1rem;
    border-radius: 10px;
    background: #fff6d8;
    border: 1px solid #efd88a;
    color: #654c00;
}
.chip {
    display: inline-block;
    padding: .25rem .65rem;
    margin: .2rem .3rem .2rem 0;
    border-radius: 999px;
    background: #e8eef8;
    font-size: .85rem;
}
.alert {
    background: #fff0f0;
    border-left: 5px solid #d62828;
    padding: .8rem 1rem;
    border-radius: 8px;
}
.safe {
    background: #eef8f1;
    border-left: 5px solid #2a9d5b;
    padding: .8rem 1rem;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<div class="header">
    <h1>🛡️ KAVACHX</h1>
    <p>Intelligent Emergency Assistance</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
<strong>Safety notice:</strong>
Do not share passwords, OTPs, PINs, card numbers, or banking credentials.
KAVACHX cannot contact emergency services, police, hospitals, or banks.
</div>
""", unsafe_allow_html=True)

st.sidebar.title("KAVACHX")
st.sidebar.markdown("""
**Supported areas**

- Women Safety
- Cyber Fraud
- Medical Emergency

Use trusted local emergency services during immediate danger.
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            metadata = message.get("metadata", {})
            st.markdown(
                f'<span class="chip">Domain: {metadata.get("domain", "Unknown")}</span>'
                f'<span class="chip">Urgency: {metadata.get("urgency", "Unknown")}</span>',
                unsafe_allow_html=True,
            )

            sources = metadata.get("sources", [])
            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"- [{source['title']}]({source['url']})"
                        )

query = st.chat_input(
    "Describe what is happening. Do not include sensitive credentials."
)

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        status = st.status("Processing request...", expanded=True)

        try:
            status.write("Analyzing intent...")
            result = run_workflow(query)

            status.write("Routing to specialist...")
            status.write("Retrieving trusted guidance...")
            status.write("Generating grounded response...")
            status.update(label="Completed", state="complete")

            answer = result["answer"]
            urgency = result["urgency"]

            if urgency in {"Critical", "High"}:
                st.markdown(
                    f'<div class="alert">{answer}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="safe">{answer}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<span class="chip">Domain: {result["domain"]}</span>'
                f'<span class="chip">Urgency: {result["urgency"]}</span>',
                unsafe_allow_html=True,
            )

            if result["sources"]:
                with st.expander("Retrieved sources"):
                    for source in result["sources"]:
                        st.markdown(
                            f"- [{source['title']}]({source['url']})"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": result,
            })

        except Exception:
            status.update(label="Unable to process safely", state="error")

            fallback = (
                "I could not safely process this request. "
                "If there is immediate danger or a serious medical emergency, "
                "contact your local emergency service now. "
                "Do not share passwords, PINs, OTPs, or banking credentials."
            )

            st.error(fallback)

            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback,
                "metadata": {
                    "domain": "Unknown",
                    "urgency": "Unknown",
                    "sources": [],
                },
            })