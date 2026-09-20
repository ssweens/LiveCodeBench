import json
import threading
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lcb_runner.lm_styles import LMStyle, LanguageModel
from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.runner.oai_runner import OpenAIRunner
from lcb_runner.utils.scenarios import Scenario


def _prompt(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


class CompletionPoolTests(unittest.TestCase):
    def setUp(self):
        self._cwd = Path.cwd()
        self._tmp = self._cwd / "cache" / "_test_completion_pool"
        self._tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        cache_dir = Path("cache") / "PoolTest"
        if cache_dir.exists():
            for path in cache_dir.glob("*.json"):
                path.unlink()

    def _args(self, *, n=5, multiprocess=3, use_cache=True):
        return SimpleNamespace(
            model="pool-test",
            temperature=0.2,
            max_tokens=128,
            top_p=0.95,
            n=n,
            openai_timeout=30,
            use_cache=use_cache,
            multiprocess=multiprocess,
            cache_batch_size=3,
            scenario=Scenario.codegeneration,
            base_url="http://127.0.0.1:9/v1",
            cot_code_execution=False,
        )

    def _model(self):
        return LanguageModel(
            "pool-test",
            "PoolTest",
            LMStyle.OpenAIChat,
            datetime(2026, 1, 1),
        )

    def _runner(self, args, create):
        with patch("lcb_runner.runner.oai_runner.OpenAI") as client_cls:
            client = client_cls.return_value
            client.chat.completions.create.side_effect = create
            runner = OpenAIRunner(args, self._model())
        runner.client.chat.completions.create.side_effect = create
        return runner

    def test_pool_keeps_multiprocess_requests_in_flight_on_one_slow_problem(self):
        in_flight = 0
        max_in_flight = 0
        started = {"slow": 0, "fast-a": 0, "fast-b": 0}
        lock = threading.Lock()
        first_wave = threading.Barrier(3, timeout=2)

        def create(*, messages, **_kwargs):
            nonlocal in_flight, max_in_flight
            content = messages[0]["content"]
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                started[content] += 1
                wave = started[content]
            if wave == 1:
                first_wave.wait()
            time.sleep(0.02 if content == "slow" else 0.001)
            with lock:
                in_flight -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

        runner = self._runner(self._args(), create)
        prompts = [_prompt("slow"), _prompt("fast-a"), _prompt("fast-b")]
        outputs = runner.run_batch(prompts, ["s", "a", "b"])
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(len(row) == 5 for row in outputs))
        self.assertEqual(max_in_flight, 3)
        self.assertEqual(started, {"slow": 5, "fast-a": 5, "fast-b": 5})

    def test_cache_is_written_when_a_problem_finishes_not_after_the_batch(self):
        cache_snapshots: list[int] = []
        lock = threading.Lock()
        holder: dict = {}

        def create(*, messages, **_kwargs):
            content = messages[0]["content"]
            time.sleep(0.08 if content == "slow" else 0.005)
            cache_path = Path(holder["runner"].cache_path)
            if cache_path.exists():
                with lock:
                    cache_snapshots.append(len(json.loads(cache_path.read_text())))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

        runner = self._runner(self._args(), create)
        holder["runner"] = runner
        prompts = [_prompt("fast-a"), _prompt("fast-b"), _prompt("slow")]
        runner.run_batch(prompts, ["a", "b", "s"])
        self.assertGreaterEqual(max(cache_snapshots or [0]), 2)
        saved = json.loads(Path(runner.cache_path).read_text())
        self.assertEqual(len(saved), 3)

    def test_serial_path_unchanged_when_multiprocess_is_off(self):
        in_flight = 0
        max_in_flight = 0
        lock = threading.Lock()

        def create(*, messages, **_kwargs):
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            time.sleep(0.01)
            with lock:
                in_flight -= 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

        runner = self._runner(self._args(multiprocess=0), create)
        self.assertFalse(runner._uses_completion_pool())
        outputs = runner.run_batch([_prompt("a"), _prompt("b")], ["a", "b"])
        self.assertEqual(len(outputs), 2)
        self.assertEqual(max_in_flight, 1)
        self.assertIs(type(runner)._run_one_completion, OpenAIRunner._run_one_completion)
        self.assertIsNot(type(runner)._run_one_completion, BaseRunner._run_one_completion)


if __name__ == "__main__":
    unittest.main()
