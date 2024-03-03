"""
Very simple example of a serverless function that just returns the input value.
"""

import requests
import runpod
import sglang as sgl

# Port of the SGLang runtime server
SGLANG_PORT = 30000
# Initialize the SGLang runtime before handling requests
RUNTIME = sgl.Runtime(model_path="/app/models/llava", port = SGLANG_PORT)
print(f"Initialized SGLang runtime: {RUNTIME.url}")

def get_model_info():
    resp = requests.get(f"http://localhost:{SGLANG_PORT}/get_model_info")
    resp.raise_for_status()
    return resp.json()

def generate(parameters: dict):
    resp = requests.post(f"http://localhost:{SGLANG_PORT}/generate", json=parameters)
    resp.raise_for_status()
    return resp.json()

def handler(job):
    print("Received job:", job)
    job_input = job["input"]
    endpoint = job_input["endpoint"]
    if endpoint == "get_model_info":
        return get_model_info()
    elif endpoint == "generate":
        return generate(job_input["parameters"])
    else:
        raise ValueError(f"Invalid endpoint `{endpoint}`.")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
