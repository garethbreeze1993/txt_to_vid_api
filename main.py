import torch
from fastapi import FastAPI, Request

from dotenv import load_dotenv

load_dotenv()  # take environment variables


from contextlib import asynccontextmanager

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from pydantic import BaseModel
from typing import TypedDict
import logging
import os
import boto3

class GenerateRequest(BaseModel):
    video_id: int
    prompt: str
    celery_task_id: str

model_id = "THUDM/CogVideoX-2b"
torch_dtype = torch.bfloat16
# quantization = int8_weight_only
local_model_dir = "/home/ubuntu/models/cogvideox"  # You can customize this path

# Define the logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(name)s: %(message)s',
        },
    },

    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'app.log',
            'formatter': 'default',
        },
    },

    'root': {
        'handlers': ['file'],
        'level': 'INFO',
    },
}

# Apply the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Example usage
logger = logging.getLogger(__name__)

class State(TypedDict):
    pipe: CogVideoXPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    app.state.pipeline = pipe
    print("Pipeline loaded and ready.")
    yield
    print("Shutting down... cleaning up.")
    del pipe


app = FastAPI(lifespan=lifespan)

@app.get("/test")
def test():
    return "PONG YOU SUNK MY BATTLESHIP"


@app.post("/generate")
def generate(request: GenerateRequest):

    pipe = app.state.pipeline

    video_id = request.video_id

    result = pipe(
        prompt=request.prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=24,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    output_path = f"/tmp/output_{video_id}.mp4"
    export_to_video(result, output_path, fps=8)

    # S3 configuration
    s3_bucket = os.environ.get("S3_BUCKET_NAME")
    s3_key = f"videos/{video_id}.mp4"

    # Upload to S3
    try:

        region = os.environ.get("AWS_REGION", "us-east-1")

        s3_client = boto3.client('s3', region_name=region)

        s3_client.upload_file(output_path, s3_bucket, s3_key)

        s3_url = f"https://{s3_bucket}.s3.{region}.amazonaws.com/{s3_key}"

        # Clean up the temporary file
        os.remove(output_path)

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    else:
        return {"message": "Video generated", "path": s3_url}
