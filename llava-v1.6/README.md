# SGLang LLava-v1.6 deployment on RunPod Serverless

This repo contains the code for deploying the LLava-v1.6 model on the RunPod serverless platform using [SGLang](https://github.com/sgl-project/sglang) for inference.

The `runpodendpoint.py` file contains a simple RunPod endpoint connector that can be used like SGLang's backend classes.

## Deployment

1. Build the docker image
    ```bash
    docker build --platform linux/amd64 -t runpod-sglang-llava-v1.6 .
    ```
    This will download multiple GB of NVIDIA's CUDA image, multiple GB of LLava-v1.6 and CLIP model weights and build the image. The final image size is around 17GB.
2. Push the image to a container registry
    ```bash
    docker tag runpod-sglang-llava-v1.6 your-registry/runpod-sglang-llava-v1.6
    docker push your-registry/runpod-sglang-llava-v1.6
    ```
    Replace `your-registry` with your actual container registry.
3. [Create a new RunPod endpoint](https://www.runpod.io/console/serverless) with the image coordinates

## Client usage

A very basic RunPod connector is included in the `runpodendpoint.py` file. It can be used like this:

```python
import sglang as sgl
from runpodendpoint import RunPodEndpoint
YOUR_RUNPOD_API_KEY = "Fill this out" # You can get this from RunPod > Settings > API Keys. Only needs READ access.
YOUR_ENDPOINT_ID = "Fill this out" # Depends on the endpoint you created. It's the last part of the endpoint dashboard URL.
runtime = RunPodEndpoint(f"https://api.runpod.ai/v2/${YOUR_ENDPOINT_ID}", YOUR_RUNPOD_API_KEY)
sgl.set_default_backend(runtime)

@sgl.function
def once_upon_a_time(s):
    s += sgl.user("Tell me a story starting with 'Once upon a time'.")
    s += sgl.assistant(sgl.gen('story', max_tokens=200))

print(once_upon_a_time.run()['story'])
```

