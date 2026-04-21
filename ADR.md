# ADR: Flexible Corpus-Grounded Legal Research Agent

## Status

Accepted

## Context

The goal of this assignment was not just to answer one narrow legal question. The agent needed to handle both straightforward corpus lookups and more open-ended precedent research over a fixed set of PDF judgments. It also needed to show its working clearly enough that someone reviewing it could understand how it got from the user’s prompt to the final answer.

That pushed me toward an approach that was general enough to handle different prompt shapes, but still simple and inspectable enough for a small corpus.

## Decision

I chose a three-part architecture: a `Planner`, a `Retriever`, and a `Synthesizer`.

### 1. Planner

The planner’s job is to decide whether the user is asking a quick corpus question or a deeper research task.

For simple prompts, the agent takes a lightweight retrieval-first path. For more research-heavy prompts, it switches into a deeper workflow and generates multiple search queries based on both the user’s wording and the likely legal issues in the prompt.

I wanted this decision to be prompt-driven rather than hard-coded to one fact pattern. In practice, the planner looks for signals like factual density, research-oriented language, and multi-part legal questions.

### 2. Retriever

The retriever ingests the PDF corpus, extracts text, creates page-aware chunks, and scores both documents and chunks with a lightweight lexical ranking approach.

I used both document-level and chunk-level retrieval because legal relevance often works at two levels at once:

- a judgment may be broadly relevant as a whole
- only a few pages may contain the holding or rule that actually matters

The retriever also extracts lightweight structured issue tags during ingestion so the system can do some issue-aware reranking instead of relying only on raw lexical overlap.

### 3. Synthesizer

The synthesizer takes the retrieved evidence, merges results across queries, and turns them into a structured research answer.

Its job is not just to summarize retrieval. It also tries to separate likely supportive precedents from adverse ones and explain why each judgment matters using the retrieved snippets. I intentionally kept the final answer grounded in corpus evidence so the reasoning is inspectable instead of feeling like a black-box memo.

## Why this architecture

This structure felt like the right balance for the assignment.

- It is flexible enough to handle both direct questions and richer research prompts without building separate hard-coded pipelines for each legal issue.
- It keeps the system transparent. The UI can show the plan, the generated searches, the retrieved documents, and the supporting passages.
- It is practical for a corpus of about 50 documents. A heavier production-style stack would have been unnecessary complexity here.
- It leaves clear upgrade paths. A stronger reranker, embeddings, richer metadata extraction, or a better synthesis layer can be added without changing the overall interface.

## Tradeoffs

I made a few deliberate tradeoffs.

- I started with lexical retrieval rather than embeddings. That made the system cheaper, faster, more deterministic, and easier to reason about, but weaker on semantic paraphrase.
- I kept the base non-LLM workflow mostly heuristic. That makes it easier to run in a constrained environment, but the final prose is less nuanced than a stronger model-backed memo writer.
- I used simple page-aware chunking rather than a more legal-document-aware segmentation strategy. It is robust and easy to implement, but not as precise as paragraph-level or citation-aware chunking.

## How the agent chooses between simple and deep workflows

The planner looks for a few high-signal patterns:

- dense factual briefing, such as client-style narratives or multiple factual details
- research-oriented words like `precedent`, `adverse`, `strategy`, `risk`, `compensation`, or `distinguish`
- prompts that are long, multi-issue, or ask multiple questions

If those signals are present, the agent switches to the deep research flow:

- query expansion
- multiple retrieval passes
- evidence merging
- structured synthesis

If not, it stays on the quick-answer path and returns a direct corpus-grounded answer from the strongest matches.

This is meant to be a general prompt-shape decision, not a special case for motor accident licence disputes.

## If the corpus were 5,000 documents instead of 50

At that size, I would change the architecture in a few important ways.

- Move PDF extraction into an offline ingestion pipeline.
- Persist normalized document records and metadata in storage rather than rebuilding in memory.
- Replace the lightweight scorer with a hybrid retrieval stack, likely BM25 or OpenSearch plus semantic retrieval.
- Add stronger metadata extraction during ingestion, especially court, year, issue tags, statutory sections, outcome, and vehicle type.
- Add a real reranking layer over top candidate chunks instead of scoring everything at query time.
- Add better observability around latency, retrieval quality, and answer quality.

The current design is appropriate for the assignment corpus, but it would not be the final production shape for a much larger collection.

## If I had another week

If I had more time, I would focus on quality rather than adding surface-level features.

- Improve retrieval with a stronger second-stage reranker and better legal issue normalization.
- Expand structured metadata extraction, especially holdings, issue tags, outcome, and insurer-liability signals.
- Improve adverse-precedent handling, since that is one of the weakest and most important areas in legal research.
- Strengthen the LLM variant so it behaves more like a grounded selector over retrieved evidence rather than a free-form summarizer.
- Add more benchmark prompts and better-labeled evaluation fixtures so recall claims are less fragile.

## Evaluation framework

The repository includes a shared evaluation runner in `evals/run_evals.py`. I designed it so both the deterministic `non_llm` agent and the `llm_variant` agent are measured against the same benchmark contract.

It evaluates the four requested dimensions:

- Precision: of the precedents the agent cites, how many are actually relevant
- Recall: of the precedents that should have been surfaced, how many the agent actually returns
- Reasoning quality: whether the explanations are grounded, issue-linked, and directionally clear
- Adverse identification: whether the agent surfaces judgments that work against the client

I also added a retrieval recall ceiling as a diagnostic metric. That is not one of the required dimensions, but it is useful because it tells me whether failures are happening during retrieval or later during classification and synthesis.

For the LLM variant, repeated runs matter because outputs can vary across samples. For the non-LLM variant, one run is usually enough because the system is deterministic.

## Current reflection

The architecture is doing what I wanted at a high level: it is flexible, inspectable, and easy to extend. The weakest part right now is not the basic planner-retriever-synthesizer split. It is the quality of retrieval and the support-versus-adverse classification logic layered on top of it.

If I were prioritizing the next improvement, I would focus first on retrieval quality and adverse-case surfacing, because those two limitations are what most directly cap downstream precision, recall, and trustworthiness.
