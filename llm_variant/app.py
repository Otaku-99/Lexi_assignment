from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "non_llm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_variant.agent import LLMResearchAgent
from llm_variant.config import detect_system_profile, recommended_ollama_model, recommended_openai_model
from llm_variant.llm_client import LLMSettings


CORPUS_DIR = ROOT / "lexi_research_take_home_assessment_docs"

DEFAULT_PROMPT = """Client: Mrs. Lakshmi Devi
Matter: Motor accident claim - death of spouse

Mrs. Lakshmi Devi's husband was killed in a road accident involving a commercial truck. The truck driver was operating the vehicle without a valid driving license at the time of the accident. The insurance company is denying liability and says the policy is void because the driver was unlicensed.

Please provide supporting precedents, adverse precedents, and a strategy recommendation."""


@st.cache_resource(show_spinner=True)
def load_agent() -> LLMResearchAgent:
    profile = detect_system_profile()
    model = recommended_ollama_model(profile)
    return LLMResearchAgent(
        corpus_dir=str(CORPUS_DIR),
        cache_dir=str(ROOT / ".cache"),
        llm_settings=LLMSettings(provider="ollama", model=model),
    )


profile = detect_system_profile()
default_ollama = recommended_ollama_model(profile)

st.set_page_config(page_title="Lexi LLM Research Agent", layout="wide")
st.title("Lexi LLM Research Agent")
st.caption("Corpus-grounded legal research over the provided Indian court judgments, with visible planning and retrieval steps.")

with st.sidebar:
    agent = load_agent()
    stats = agent.stats()
    st.subheader("Corpus")
    st.write(f"Documents: `{stats['documents']}`")
    st.write(f"Chunks: `{stats['chunks']}`")
    st.write(f"Cache: `{stats['cache_path']}`")
    st.markdown(
        "Try both quick queries like `Which judgments involve commercial vehicles?` and deeper research prompts asking for precedents, risk, and strategy."
    )

prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=260)

if st.button("Run Research", type="primary", use_container_width=True):
    agent = load_agent()
    with st.spinner("Planning, retrieving, and calling the selected model..."):
        try:
            response = agent.run(prompt)
            left, right = st.columns([1.35, 1])
            with left:
                st.markdown(response.answer_markdown)
            with right:
                st.subheader("Trace")
                st.json(
                    {
                        "mode": response.mode,
                        "plan": {
                            "rationale": response.plan.rationale,
                            "issue_tags": response.plan.issue_tags,
                            "search_queries": response.plan.search_queries,
                        },
                        "trace": response.trace,
                    }
                )

            st.subheader("Retrieved Evidence")
            for result in response.search_results:
                with st.expander(f"Query: {result.query}"):
                    st.markdown("**Top documents**")
                    for doc in result.documents[:6]:
                        st.markdown(
                            f"- `{doc.file_name}` | **{doc.title}** | score `{doc.score:.3f}` | matched `{', '.join(doc.matched_terms) or 'n/a'}`"
                        )
                        st.caption(doc.snippet)
                    st.markdown("**Top chunks**")
                    for chunk in result.chunks[:8]:
                        st.markdown(
                            f"- `{chunk.file_name}` pages `{chunk.page_start}-{chunk.page_end}` | score `{chunk.score:.3f}` | matched `{', '.join(chunk.matched_terms) or 'n/a'}`"
                        )
                        st.caption(chunk.snippet)
        except Exception as exc:
            st.error(str(exc))
            st.info(f"Make sure Ollama is running and the default model `{default_ollama}` has been pulled.")
