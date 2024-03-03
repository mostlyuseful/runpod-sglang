from typing import List

import numpy as np
import requests
from sglang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams


class RunPodEndpoint(BaseBackend):
    """RunPod serverless endpoint backend. This backend is closely based on `sglang.backend.runtime_endpoint.RuntimeEndpoint`.
    A RunPod serverless endpoint is expected to take a payload like this:
    {
        "input": ...
    }

    and return a response like this:
    {
        "output": ...
    }
    Since RunPod manages the endpoints, there is only one URL path to communicate with. This backend connector wraps
    the different "endpoints" to JSON payloads that are sent to the RunPod serverless endpoint.

    For example:

    curl http://localhost:30000/generate \                                                                                                                          ✔  moe@bigrig 
    -H "Content-Type: application/json" \
    -d '{
        "text": "Once upon a time,",                                    
        "sampling_params": {
        "max_new_tokens": 200,
        "temperature": 0  
        }
    }'

    is in the RunPod deployment equivalent to:

    curl -v --request POST https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync \
    -H "Authorization: Bearer ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "endpoint": "generate",
            "parameters": {
                "text": "Once upon a time,",
                "sampling_params": {
                    "max_new_tokens": 200,
                    "temperature": 0
                }
            }
        }
    }'

    Beware that some operations may take quite a long time to complete, and the RunPod serverless endpoint may seem to be unresponsive.
    In this case, check your worker allocation budget, RunPod logs and metrics.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        """Create a new RunPod endpoint connection.
        
        :param base_url: The base URL of the RunPod serverless endpoint, e.g. "https://api.runpod.ai/v2/abcdefg"
        :param api_key: The RunPod API key. You can generate one from Settings > API Keys > + API Key. It only needs READ access.
        """
        super().__init__()
        self.support_concate_and_append = False # What is this setting for?
        self.base_url = base_url
        self.session = requests.Session()

        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        # Get model info
        resp = self.session.post(
            f"{self.base_url}/runsync", json={"input": {"endpoint": "get_model_info"}}
        )
        resp.raise_for_status()
        self.model_info = resp.json()["output"]
        self.chat_template = get_chat_template_by_model_path(
            self.model_info["model_path"]
        )

    def get_model_name(self):
        return self.model_info["model_path"]

    def get_chat_template(self):
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={
                "input": {
                    "endpoint": "generate",
                    "parameters": {
                        "text": prefix_str,
                        "sampling_params": {"max_new_tokens": 0},
                    },
                }
            },
        )
        resp.raise_for_status()

    def commit_lazy_operations(self, s: StreamExecutor):
        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={
                "input": {
                    "endpoint": "generate",
                    "parameters": {
                        "text": s.text_,
                        "sampling_params": {"max_new_tokens": 0},
                    },
                }
            },
        )
        resp.raise_for_status()

    def fill_image(self, s: StreamExecutor):
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={"input": {"endpoint": "generate", "parameters": data}},
        )
        resp.raise_for_status()

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if sampling_params.dtype is None:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    **sampling_params.to_srt_kwargs(),
                },
            }
        elif sampling_params.dtype in [int, "int"]:
            data = {
                "text": s.text_,
                "sampling_params": {
                    "skip_special_tokens": global_config.skip_special_tokens_in_output,
                    "dtype": "int",
                    **sampling_params.to_srt_kwargs(),
                },
            }
        else:
            raise RuntimeError(f"Invalid dtype: {sampling_params.dtype}")

        self._add_images(s, data)

        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={"input": {"endpoint": "generate", "parameters": data}},
        )
        resp.raise_for_status()
        obj = resp.json()["output"]
        comp = obj["text"]
        return comp, obj["meta_info"]

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        raise NotImplementedError("Not supported")

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
    ):
        assert temperature <= 1e-5

        # Cache common prefix
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={"input": {"endpoint": "generate", "parameters": data}},
        )
        resp.raise_for_status()
        prompt_len = resp.json()["output"]["meta_info"]["prompt_tokens"]

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {"max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": max(prompt_len - 2, 0),
        }
        self._add_images(s, data)
        resp = self.session.post(
            f"{self.base_url}/runsync",
            json={"input": {"endpoint": "generate", "parameters": data}},
        )
        resp.raise_for_status()
        obj = resp.json()["output"]
        normalized_prompt_logprob = [
            r["meta_info"]["normalized_prompt_logprob"] for r in obj
        ]
        prompt_logprob = [r["meta_info"]["prompt_logprob"] for r in obj]

        decision = choices[np.argmax(normalized_prompt_logprob)]
        return decision, normalized_prompt_logprob, prompt_logprob

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        raise NotImplementedError("Not supported")

    def _add_images(self, s: StreamExecutor, data):
        if s.images_:
            assert len(s.images_) == 1, "Only support one image."
            data["image_data"] = s.images_[0][1]
