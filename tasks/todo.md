# LiveCodeBench Local LLM Optimization

## Tasks
- [ ] Investigate and fix `APITimeoutError` in `OpenAIRunner` <!-- id: 0 -->
    - [ ] Check `lcb_runner/runner/parser.py` for `--openai_timeout` default and usage <!-- id: 1 -->
    - [ ] Check `lcb_runner/runner/oai_runner.py` to see if timeout is passed to the client <!-- id: 2 -->
    - [ ] Increase default timeout if necessary <!-- id: 3 -->
- [ ] Optimize concurrency/multiprocessing <!-- id: 4 -->
    - [ ] Investigate how `--multiprocess` affects `OpenAIRunner` <!-- id: 5 -->
    - [ ] Recommend or implement better default concurrency for local servers <!-- id: 6 -->
- [ ] Final verification of "Rich Evaluation" and "Resume" logic <!-- id: 7 -->
- [ ] Quality gates (tests, linters) <!-- id: 8 -->
- [ ] Commit and push changes <!-- id: 9 -->

## Lessons Learned
- Python 3.12 is required for `vllm`/`ray` compatibility.
- Local models should be auto-registered as OpenAI-compatible.
- Evaluation metrics keys should be normalized to strings for JSON compatibility.
