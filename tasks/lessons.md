# Lessons Learned

- [2025-03-24] Always verify external resources (datasets, APIs, libraries) exist and are accessible before writing code that depends on them. Context: Wrote dataset loader plan based on assumed HuggingFace datasets, but 7/11 text datasets and 7/8 vision datasets failed verification due to legacy scripts, gated access, or non-existence.
- [2026-03-30] Validate multi-device JAX behavior with a minimal runtime repro before assuming a NamedSharding regression. Context: A suspected sharding-loss theory did not reproduce; the real training risks were in LR scheduling inside JIT and streaming retry byte accounting.
