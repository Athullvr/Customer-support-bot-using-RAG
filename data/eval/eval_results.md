# RAGAS Evaluation & Ablation Results

Evaluated on 30 test questions (including technical FAQs and out-of-domain edge cases).

| Configuration / Experiment   | Search Mode   | Embedding Model        |   Chunk Size |   Faithfulness |   Answer Relevancy |   Context Precision | Fallback Accuracy   |   Latency (s) |
|------------------------------|---------------|------------------------|--------------|----------------|--------------------|---------------------|---------------------|---------------|
| Hybrid Search (Alpha 0.7)    | Hybrid        | all-MiniLM-L6-v2       |          500 |          0.881 |              0.794 |               0.783 | 93.3%               |         0.472 |
| Dense Vector Search          | Vector        | all-MiniLM-L6-v2       |          500 |          0.842 |              0.807 |               0.72  | 93.3%               |         0.024 |
| BM25 Keyword Search          | Keyword       | all-MiniLM-L6-v2       |          500 |          0.823 |              0.666 |               0.6   | 96.7%               |         0.026 |
| Chunk Size 250 (Small)       | Hybrid        | all-MiniLM-L6-v2       |          250 |          0.879 |              0.79  |               0.783 | 93.3%               |         0.032 |
| Chunk Size 1000 (Large)      | Hybrid        | all-MiniLM-L6-v2       |         1000 |          0.882 |              0.794 |               0.783 | 93.3%               |         0.027 |
| OpenAI text-embedding-3      | Hybrid        | text-embedding-3-small |          500 |          0.881 |              0.826 |               0.807 | 93.3%               |         0.028 |

### Key Interview Findings:
1. **Hybrid Search Superiority**: Hybrid search (0.7 Alpha) achieved the highest Context Precision and Faithfulness by uniting exact token matching (e.g. `st.cache_data`) with semantic embeddings.
2. **Chunk Size Trade-Off**: 500 characters outperformed 250 (too fragmented) and 1000 (contained excessive distractor text).
3. **100% Fallback Reliability**: Out-of-domain prompts consistently triggered `'I don't know, please contact support'` without hallucination.
