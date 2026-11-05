# frontend/streamlit_app.py
import streamlit as st
import requests

st.set_page_config(page_title="SciDoc RAG Assistant", layout="wide", page_icon="🧬")

st.title("🔬 Scientific Document Question Answering")
st.caption("Retrieval-Augmented Generation over the SciFact Corpus")

# Sidebar configurations
with st.sidebar:
    st.header("⚙️ System Controls")
    retrieval_method = st.selectbox(
        "Retrieval Engine",
        ["Sentence-BERT (all-MiniLM-L6-v2)", "multi-qa-MiniLM", "TF-IDF Baseline"]
    )
    top_k = st.slider("Documents to Retrieve (Top-K)", min_value=1, max_value=5, value=3)
    backend_url = st.text_input("Backend API URL", "http://localhost:8000")
    
    st.divider()
    st.markdown("**Project Team:**")
    st.markdown("- Saba Noor (Team Lead)\n- Laiba Shaikh\n- Najaf")

# Main Query Area
query = st.text_input(
    "Enter a scientific question or biomedical claim:",
    placeholder="e.g., Do statins reduce the incidence of stroke?"
)

if st.button("Search & Generate Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving evidence and synthesizing answer..."):
            try:
                payload = {
                    "query": query,
                    "method": retrieval_method,
                    "top_k": top_k
                }
               
                res = requests.post(f"{backend_url}/api/ask", json=payload, timeout=60)
                
                if res.status_code == 200:
                    data = res.json()
                    
                    col1, col2 = st.columns([1.2, 1])
                    
                    with col1:
                        st.subheader("💡 Generated Answer")
                        st.markdown(data["answer"])
                        st.info(f"Engine used: **{data['method']}** | Evidence count: **{len(data['sources'])}**")
                        
                    with col2:
                        st.subheader("📄 Retrieved Evidence (SciFact)")
                        for idx, doc in enumerate(data["sources"], 1):
                            with st.expander(f"[{doc['id']}] {doc['title']} (Score: {doc['score']:.2f})", expanded=(idx==1)):
                                st.write(doc["text"])
                                st.caption(f"Doc ID: {doc['id']}")
                else:
                    st.error(f"Backend error: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")