import os
from time import sleep

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    pass

from lcb_runner.lm_styles import LMStyle
from lcb_runner.runner.base_runner import BaseRunner

# Default to corral's local server port; override with --base-url or OPENAI_BASE_URL
_DEFAULT_BASE_URL = "http://localhost:9999/v1"


class OpenAIRunner(BaseRunner):

    def __init__(self, args, model):
        super().__init__(args, model)

        # Resolve base_url: CLI flag > env var > default (corral port 9999)
        base_url = getattr(args, "base_url", None)
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL", _DEFAULT_BASE_URL)

        api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        if model.model_style == LMStyle.OpenAIReasonPreview:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "max_completion_tokens": 25000,
            }
        elif model.model_style == LMStyle.OpenAIReason:
            assert (
                "__" in args.model
            ), f"Model {args.model} is not a valid OpenAI Reasoning model as we require reasoning effort in model name."
            model, reasoning_effort = args.model.split("__")
            self.client_kwargs: dict[str | str] = {
                "model": model,
                "reasoning_effort": reasoning_effort,
            }
        else:
            self.client_kwargs: dict[str | str] = {
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": args.n,
                "timeout": args.openai_timeout,
                # "stop": args.stop, --> stop is only used for base models currently
            }

    def _run_single(self, prompt: list[dict[str, str]], retries: int = 10) -> list[str]:
        assert isinstance(prompt, list)

        n_completions = self.client_kwargs.get("n", 1)
        # Always request n=1 per call (compatible with single-slot servers like
        # llama.cpp) and loop to collect the desired number of completions.
        call_kwargs = {**self.client_kwargs, "n": 1}

        results: list[str] = []
        for _ in range(n_completions):
            result = self._call_once(prompt, call_kwargs, retries=retries)
            results.append(result)
        return results

    def _call_once(self, prompt: list[dict[str, str]], call_kwargs: dict, retries: int = 10) -> str:
        if retries == 0:
            print("Max retries reached. Returning empty response.")
            return ""

        try:
            response = self.client.chat.completions.create(
                messages=prompt,
                **call_kwargs,
            )
        except (
            openai.APIError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.OpenAIError,
            openai.APIStatusError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self._call_once(prompt, call_kwargs, retries=retries - 1)
        except Exception as e:
            print(f"Failed to run the model for {prompt}!")
            print("Exception: ", repr(e))
            raise e
        return response.choices[0].message.content
