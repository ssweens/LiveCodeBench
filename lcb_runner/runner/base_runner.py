import json
import os
import sys
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from lcb_runner.lm_styles import LanguageModel
from lcb_runner.utils.path_utils import get_cache_path
from lcb_runner.utils.multiprocess import run_tasks_in_parallel
from lcb_runner.runner.scenario_router import Scenario


# Reserved cache value for a completion that could not be obtained because of
# a request/process failure. Empty strings remain valid completed model answers.
# Garage's cache sanitizer recognizes this value and removes its whole prompt
# entry instead of scoring it as a model failure.
REQUEST_FAILURE_SENTINEL = "__GARAGE_LCB_TRANSPORT_FAILURE_V1__"


def _prompt_cache_key(prompt: str | list[dict[str, str]] | tuple) -> str:
    if isinstance(prompt, list):
        return json.dumps(prompt)
    if isinstance(prompt, tuple):
        return prompt[0] + json.dumps(prompt[1])
    return prompt


class BaseRunner(ABC):
    def __init__(self, args, model: LanguageModel):
        self.args = args
        self.model = model
        self.client_kwargs: dict[str | str] = {}
        self._cache_lock = threading.RLock()
        self._progress_lock = threading.Lock()

        if self.args.use_cache:
            self.cache_path = get_cache_path(model.model_repr, args)
            if os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    self.cache: dict = json.load(f)
            else:
                self.cache = {}
        else:
            self.cache_path = None
            self.cache = None

        progress_path = os.environ.get("GARAGE_LCB_PROGRESS_PATH")
        self._progress_path = Path(progress_path) if progress_path else None
        self._progress_question_ids: list[str] = []
        self._progress_total = 0
        self._progress_completed = 0
        self._progress_current_index: int | None = None
        self._progress_current_question_id: str | None = None
        self._progress_current_try = 0

    def _write_progress(self) -> None:
        if self._progress_path is None:
            return
        payload = {
            "schema_version": 1,
            "total_problems": self._progress_total,
            "completed_problems": self._progress_completed,
            "tries_per_problem": self.args.n,
            "current_index": self._progress_current_index,
            "current_question_id": self._progress_current_question_id,
            "current_try": self._progress_current_try,
            "updated_at": time.time(),
        }
        self._progress_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self._progress_path.name}.",
            suffix=".tmp",
            dir=self._progress_path.parent,
        )
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._progress_path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def _progress_start(self, question_ids: list[str]) -> None:
        if self._progress_path is None:
            return
        self._progress_question_ids = question_ids
        self._progress_total = len(question_ids)
        self._progress_completed = 0
        self._progress_current_index = None
        self._progress_current_question_id = None
        self._progress_current_try = 0
        self._write_progress()

    def _progress_begin(self, question_id: str) -> None:
        if self._progress_path is None:
            return
        with self._progress_lock:
            self._progress_current_index = self._progress_question_ids.index(question_id)
            self._progress_current_question_id = question_id
            self._progress_current_try = 0
            self._write_progress()

    def _progress_try(self, completed_try: int) -> None:
        if self._progress_path is None:
            return
        with self._progress_lock:
            if self._progress_current_index is None:
                return
            self._progress_current_try = completed_try
            self._write_progress()

    def _progress_complete(self) -> None:
        if self._progress_path is None:
            return
        with self._progress_lock:
            self._progress_completed += 1
            self._progress_current_index = None
            self._progress_current_question_id = None
            self._progress_current_try = 0
            self._write_progress()

    def save_cache(self):
        if not self.args.use_cache:
            return
        with self._cache_lock:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=4)

    # @abstractmethod
    def _run_single(self, prompt: str | list[dict[str, str]]) -> list[str]:
        pass

    def _run_one_completion(self, prompt: str | list[dict[str, str]]) -> str:
        """Generate a single completion. OpenAI-style runners override this so
        the completion pool can keep `--multiprocess` HTTP calls in flight
        across problems instead of serializing n tries inside one problem.
        """
        raise NotImplementedError

    def _uses_completion_pool(self) -> bool:
        return (
            self.args.multiprocess > 1
            and type(self)._run_one_completion is not BaseRunner._run_one_completion
        )

    def _cached_slots(self, prompt: str | list[dict[str, str]] | tuple) -> list[str | None]:
        n = self.args.n
        slots: list[str | None] = [None] * n
        if self.cache is None:
            return slots
        cached = self.cache.get(_prompt_cache_key(prompt))
        if not isinstance(cached, list):
            return slots
        for index, value in enumerate(cached[:n]):
            if value is None or value == REQUEST_FAILURE_SENTINEL:
                continue
            slots[index] = "" if value is None else str(value)
        return slots

    @staticmethod
    def run_single(combined_args) -> list[str]:
        """
        Run the model for a single prompt and return the output
        Static method to be used in multiprocessing
        Calls the _run_single method with the combined arguments
        """
        prompt: str | list[dict[str, str]]
        cache: dict[str, str]
        call_method: callable
        prompt, cache, args, call_method = combined_args

        if isinstance(prompt, list):
            prompt_cache = json.dumps(prompt)
        elif isinstance(prompt, tuple):
            prompt_cache = prompt[0] + json.dumps(prompt[1])
        else:
            prompt_cache = prompt      

        if cache is not None and prompt_cache in cache:
            if len(cache[prompt_cache]) == args.n:
                return ["" if x is None else str(x) for x in cache[prompt_cache]]

        result = call_method(prompt)
        assert len(result) == args.n
        result = ["" if x is None else str(x) for x in result]

        return result

    def run_batch(
        self,
        prompts: list[str | list[dict[str, str]]],
        progress_items: list[str] | None = None,
    ) -> list[list[str]]:
        if self._uses_completion_pool():
            return self._run_batch_completion_pool(prompts, progress_items)
        return self._run_batch_per_prompt(prompts, progress_items)

    def _run_batch_per_prompt(
        self,
        prompts: list[str | list[dict[str, str]]],
        progress_items: list[str] | None = None,
    ) -> list[list[str]]:
        outputs = []
        arguments = [
            (
                prompt,
                self.cache,  ## pass the cache as argument for cache check
                self.args,  ## pass the args as argument for cache check
                self._run_single,  ## pass the _run_single method as argument because of multiprocessing
            )
            for prompt in prompts
        ]
        if self.args.multiprocess > 1:
            def _process_item(arg):
                try:
                    return self.run_single(arg)
                except Exception as exc:
                    print(f"Failed to run the model for prompt: {exc}")
                    return [REQUEST_FAILURE_SENTINEL] * self.args.n

            with ThreadPoolExecutor(max_workers=self.args.multiprocess) as executor:
                parallel_results = list(
                    tqdm(
                        executor.map(_process_item, arguments),
                        total=len(arguments),
                        desc="Generating completions",
                        dynamic_ncols=True,
                        file=sys.stdout,
                    )
                )

            for result in parallel_results:
                outputs.append(result)

            if progress_items is not None:
                for question_id in progress_items:
                    self._progress_begin(question_id)
                    self._progress_complete()
        else:
            for index, argument in enumerate(tqdm(arguments)):
                if progress_items is not None:
                    self._progress_begin(progress_items[index])
                outputs.append(self.run_single(argument))
                if progress_items is not None:
                    self._progress_complete()

        if self.args.use_cache:
            for prompt, output in zip(prompts, outputs):
                self.cache[_prompt_cache_key(prompt)] = output

        return outputs

    def _run_batch_completion_pool(
        self,
        prompts: list[str | list[dict[str, str]]],
        progress_items: list[str] | None = None,
    ) -> list[list[str]]:
        n = self.args.n
        workers = max(1, int(self.args.multiprocess))
        outputs: list[list[str | None]] = []
        remaining = []
        jobs: list[tuple[int, int]] = []

        slot_lists: list[list[int]] = []
        for prompt_index, prompt in enumerate(prompts):
            slots = self._cached_slots(prompt)
            outputs.append(slots)
            missing = [slot_index for slot_index, value in enumerate(slots) if value is None]
            remaining.append(len(missing))
            slot_lists.append(missing)
            if not missing and progress_items is not None:
                self._progress_begin(progress_items[prompt_index])
                self._progress_complete()
        max_missing = max((len(missing) for missing in slot_lists), default=0)
        for offset in range(max_missing):
            for prompt_index, missing in enumerate(slot_lists):
                if offset < len(missing):
                    jobs.append((prompt_index, missing[offset]))

        if not jobs:
            return [[str(value) for value in row] for row in outputs]

        def _work(job: tuple[int, int]) -> None:
            prompt_index, slot_index = job
            prompt = prompts[prompt_index]
            try:
                result = self._run_one_completion(prompt)
            except Exception as exc:
                print(f"Failed to run the model for prompt: {exc}")
                result = REQUEST_FAILURE_SENTINEL
            result = "" if result is None else str(result)
            with self._cache_lock:
                outputs[prompt_index][slot_index] = result
                remaining[prompt_index] -= 1
                filled = n - remaining[prompt_index]
                if progress_items is not None:
                    self._progress_begin(progress_items[prompt_index])
                    self._progress_try(filled)
                    if remaining[prompt_index] == 0:
                        self._progress_complete()
                if remaining[prompt_index] == 0 and self.args.use_cache:
                    self.cache[_prompt_cache_key(prompt)] = [
                        "" if value is None else str(value) for value in outputs[prompt_index]
                    ]
                    self.save_cache()

        desc = "Generating completions"
        if workers == 1:
            for job in tqdm(jobs, total=len(jobs), desc=desc, dynamic_ncols=True, file=sys.stdout):
                _work(job)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_work, job) for job in jobs]
                for _ in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=desc,
                    dynamic_ncols=True,
                    file=sys.stdout,
                ):
                    pass
                for future in futures:
                    future.result()

        return [["" if value is None else str(value) for value in row] for row in outputs]

    def prompts_to_outputs(
        self,
        prompts: list[str | list[dict[str, str]]],
        progress_items: list[str] | None = None,
    ) -> list[list[str]]:
        if progress_items is not None:
            assert len(progress_items) == len(prompts)
            self._progress_start(progress_items)
        if self._uses_completion_pool():
            outputs = self.run_batch(prompts, progress_items)
            if self.args.use_cache:
                self.save_cache()
            return outputs
        if self.args.use_cache:
            outputs = []
            batch_size = self.args.cache_batch_size
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                items = progress_items[i : i + batch_size] if progress_items is not None else None
                batch_outputs = self.run_batch(batch, items)
                outputs.extend(batch_outputs)
                self.save_cache()
            return outputs
        return self.run_batch(prompts, progress_items)

    def run_main_repair(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        assert self.args.n == 1
        with open(
            f"output/{self.model.model_repr}/{Scenario.codegeneration}_{self.args.codegen_n}_{self.args.temperature}_eval_all.json"
        ) as f:
            check_metadata_list = json.load(f)

        outputs = [
            [None for _ in range(self.args.codegen_n)]
            for _ in range(len(benchmark))
        ]
        prompts = []
        prompt_index_to_question_idx = {}
        prompt_index_to_code_idx = {}
        count = 0

        for problem_idx, problem in enumerate(benchmark):
            for check_metadata_idx, check_metadata in enumerate(check_metadata_list):
                if problem.question_id == check_metadata['question_id']:
                    count += 1 
                    question_content = check_metadata["question_content"]
                    code_list = check_metadata["code_list"]
                    output_list = check_metadata["output_list"]
                    graded_list = check_metadata["graded_list"]
                    metadata = check_metadata["metadata"]
                    for code_idx in range(len(code_list)):
                        prompt = format_prompt(
                            question_content,
                            self.model.model_style,
                            code_list[code_idx],
                            graded_list[code_idx],
                            metadata[code_idx],
                        )
                        if prompt == "":
                            outputs[problem_idx][code_idx] = output_list[code_idx]
                            continue
                        prompts.append(prompt)
                        prompt_index_to_question_idx[len(prompts) - 1] = problem_idx
                        prompt_index_to_code_idx[len(prompts) - 1] = code_idx

        assert len(benchmark)==count, f"{len(benchmark)=}!={count=}"

        prompt_outputs = self.prompts_to_outputs(prompts)
        for prompt_idx, output in enumerate(prompt_outputs):
            question_idx = prompt_index_to_question_idx[prompt_idx]
            code_idx = prompt_index_to_code_idx[prompt_idx]
            outputs[question_idx][code_idx] = output

        return outputs

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        if self.args.scenario == Scenario.selfrepair:
            return self.run_main_repair(benchmark, format_prompt)

        prompts = [
            format_prompt(problem, self.model.model_style) for problem in benchmark
        ]
        outputs = self.prompts_to_outputs(
            prompts, [str(problem.question_id) for problem in benchmark]
        )
        return outputs
