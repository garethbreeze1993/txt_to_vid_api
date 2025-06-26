import torch
from fastapi import FastAPI, Request, BackgroundTasks
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()  # take environment variables


from contextlib import asynccontextmanager

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from pydantic import BaseModel
from typing import TypedDict, Optional
import logging
import os
import boto3
import requests

from database import JobManager

class GenerateRequest(BaseModel):
    video_id: int
    prompt: str
    celery_task_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    message: Optional[str] = None
    video_url: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

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

    JobManager.init_db()

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    app.state.pipeline = pipe
    logger.debug(f"Pipeline loaded and ready. at --- {datetime.now()}")
    yield
    logger.debug(f"Shutting down... cleaning up. at --- {datetime.now()}")
    del pipe


app = FastAPI(lifespan=lifespan)


def notify_django_completion(video_id: int, job_id: str, status: str, video_url: str = None, error_message: str = None):
    """Notify Django site about job completion"""

    if not os.environ.get("DJANGO_WEBHOOK_URL"):
        logger.error(f"No env variable for webhook url")
        return

    payload = {
        "video_id": video_id,
        "job_id": job_id,
        "status": status,
        "completed_at": datetime.now().isoformat(),
        "video_url": video_url,
        "error_message": error_message
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "VideoGen-Service/1.0"
    }

    # Add API key if configured
    if os.environ.get("DJANGO_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ.get("DJANGO_API_KEY")}"

    else:
        logger.error(f"No env variable for api key")
        return

    try:
        response = requests.post(url=f"{os.environ.get("DJANGO_WEBHOOK_URL")}", json=payload, headers=headers)

        if response.status_code == 200:
            logger.debug(f"✓ Django notified successfully for job {job_id}")
        else:
            logger.error(f"⚠ Django notification failed for job {job_id}: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Error notifying Django for job {job_id}: {str(e)}")


def background_generate(job_id: str, prompt: str, video_id: int):

    JobManager.update_job(job_id, {
        "status": "processing",
        "message": "Generating video..."
    })

    pipe = app.state.pipeline

    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=24,
        guidance_scale=6,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).frames[0]

    JobManager.update_job(job_id, {
        "status": "processing",
        "message": "Uploading to S3..."
    })

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
        JobManager.update_job(job_id, {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "completed_at": datetime.now().isoformat()
        })

        notify_django_completion(video_id, job_id, "failed", error_message=str(e))

    else:
        # Update job status to completed
        JobManager.update_job(job_id, {
            "status": "completed",
            "video_url": s3_url,
            "message": "Video generated successfully",
            "completed_at": datetime.now().isoformat()
        })

        notify_django_completion(video_id, job_id, "completed", video_url=s3_url)


@app.get("/test")
def test():
    return "PONG YOU SUNK MY BATTLESHIP"

@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    # Create job in database
    JobManager.create_job(request.celery_task_id, request.prompt, request.video_id)

    background_tasks.add_task(

        background_generate,
        request.celery_task_id,
        request.prompt,
        request.video_id
    )

    return {
        "job_id": request.celery_task_id,
        "status": "pending",
        "message": "Video generation started. Use the job_id to check status."
    }


@app.get("/status/{job_id}")
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a video generation job"""

    job_data = JobManager.get_job(job_id)

    if not job_data:
        return JobStatusResponse(
            job_id=job_id,
            status="not_found",
            message="Job not found or expired",
            created_at=datetime.now().isoformat()
        )

    return JobStatusResponse(**job_data)