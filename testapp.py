from fastapi import FastAPI, Request, BackgroundTasks
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # take environment variables


from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import TypedDict
import logging
import os

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    JobManager.init_db()
    yield
app = FastAPI(lifespan=lifespan)


def background_generate(job_id: str, prompt: str, video_id: int):
    JobManager.update_job(job_id, {
        "status": "completed",
        "video_url": "google.com",
        "message": "Video generated successfully",
        "completed_at": datetime.now().isoformat()
    })
    return {"message": "Video generated"}

@app.get("/test")
def test():
    return "PONG YOU SUNK MY BATTLESHIP"

@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
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