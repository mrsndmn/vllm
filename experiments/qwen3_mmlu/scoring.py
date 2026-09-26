"""Score single-token choices using vLLM's unmodified next-token logprobs."""

import math

import requests as http
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI


@register_model("local-single-token")
class SingleTokenAPI(LocalCompletionsAPI):
    """Avoid echoed prompt logits while retaining the harness likelihood metric.

    vLLM returns raw logprobs before applying allowed_token_ids. Forcing one
    candidate therefore exposes its true logprob and allows prefix-cache reuse.
    This adapter deliberately rejects multi-token targets and context truncation.
    """

    def _create_payload(self, messages, *, generate=False, **kwargs):
        if generate:
            raise ValueError("This adapter supports single-token likelihoods only")
        if len(messages) != 1 or not isinstance(messages[0], list):
            raise ValueError("Use tokenized requests and batch_size=1")
        tokens = messages[0]
        return {
            "model": self.model,
            "prompt": [tokens[:-1]],
            "allowed_token_ids": [tokens[-1]],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": 1,
            "seed": kwargs.get("seed", 1234),
        }

    @staticmethod
    def parse_logprobs(outputs, tokens=None, ctxlens=None, **kwargs):
        if isinstance(outputs, list) and len(outputs) == 1:
            outputs = outputs[0]
        if not isinstance(outputs, dict) or len(outputs.get("choices", [])) != 1:
            raise ValueError("Expected one completion with raw log probabilities")
        choice = outputs["choices"][0]
        values = choice["logprobs"]["token_logprobs"]
        if len(values) != 1 or values[0] is None:
            raise ValueError("Expected exactly one scored output token")
        best = max(choice["logprobs"]["top_logprobs"][0].values())
        return [(values[0], values[0] == best)]

    def _loglikelihood_tokens(self, requests, **kwargs):
        if self._batch_size != 1 or not self.tokenized_requests:
            raise ValueError("Use tokenized requests and batch_size=1")
        for _, context, continuation in requests:
            if len(continuation) != 1 or not context:
                raise ValueError("Each choice must be exactly one token after context")
            if len(context) + 1 > self.max_length:
                raise ValueError(
                    "Context exceeds configured limit; refusing truncation"
                )
        self._verify_raw_scores(requests[:4])
        return super()._loglikelihood_tokens(requests, **kwargs)

    def _verify_raw_scores(self, requests):
        errors = []
        for _, context, continuation in requests:
            tokens = [context + continuation]
            forced = self.model_call(tokens, generate=False)
            score = self.parse_logprobs(forced)[0][0]
            payload = LocalCompletionsAPI._create_payload(self, tokens, generate=False)
            response = http.post(
                self.base_url,
                json=payload,
                headers=self.header,
                timeout=self.timeout,
            )
            response.raise_for_status()
            expected = LocalCompletionsAPI.parse_logprobs(
                response.json(), ctxlens=[len(context)]
            )[0][0]
            if not math.isfinite(score) or abs(score - expected) > 0.05:
                raise ValueError(
                    f"Raw-logprob check failed: forced={score}, echoed={expected}"
                )
            errors.append(abs(score - expected))
        print(
            f"Raw-logprob preflight: max error {max(errors, default=0):.6f}", flush=True
        )


if __name__ == "__main__":
    cli_evaluate()
