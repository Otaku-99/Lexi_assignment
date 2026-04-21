from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_agent import LegalResearchAgent


CORPUS_DIR = PROJECT_ROOT / "lexi_research_take_home_assessment_docs"

DEFAULT_PROMPT = """Client: Mrs. Lakshmi Devi
Matter: Motor accident claim — death of spouse

Mrs. Lakshmi Devi's husband was killed in a road accident involving a commercial truck. The truck driver was operating the vehicle without a valid driving license at the time of the accident. The insurance company (National Insurance Co.) is denying the claim on the ground that the motor insurance policy is void due to the driver being unlicensed, and therefore they bear no liability to pay compensation.

Key Facts:
- The deceased was 42 years old at the time of the accident
- Monthly income: ₹35,000
- Dependents: wife (Mrs. Lakshmi Devi) and two minor children (ages 8 and 12)
- The truck was a commercial vehicle owned by a transport company
- The truck driver did not hold a valid driving license
- The insurance company is contesting liability, arguing the policy is void

Please provide supporting precedents, adverse precedents, and a strategy recommendation."""


@st.cache_resource(show_spinner=True)
def load_agent() -> LegalResearchAgent:
    return LegalResearchAgent(corpus_dir=str(CORPUS_DIR), cache_dir=str(PROJECT_ROOT / ".cache"))


st.set_page_config(page_title="Lexi Legal Research Agent", layout="wide")
st.title("Lexi Legal Research Agent")
st.caption("Corpus-grounded research over the provided Indian court judgments, with visible planning and retrieval steps.")

agent = load_agent()
stats = agent.stats()

with st.sidebar:
    st.subheader("Corpus")
    st.write(f"Documents: `{stats['documents']}`")
    st.write(f"Chunks: `{stats['chunks']}`")
    st.write(f"Cache: `{stats['cache_path']}`")
    st.markdown(
        "Try both quick queries like `Which judgments involve commercial vehicles?` and deeper research prompts asking for precedents, risk, and strategy."
    )

prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=320)

if st.button("Run Research", type="primary", use_container_width=True):
    with st.spinner("Planning, retrieving, and synthesizing from the corpus..."):
        response = agent.run(prompt)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(response.answer_markdown)
    with right:
        st.subheader("Reasoning Trace")
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
            for doc in result.documents:
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
