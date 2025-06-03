import torch
from fastapi import FastAPI, Request

from dotenv import load_dotenv

load_dotenv()  # take environment variables


from contextlib import asynccontextmanager

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from typing import TypedDict
import os
import boto3

model_id = "THUDM/CogVideoX-2b"
torch_dtype = torch.bfloat16
# quantization = int8_weight_only
local_model_dir = "/home/ubuntu/models/cogvideox"  # You can customize this path

class State(TypedDict):
    pipe: CogVideoXPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # print("Starting up... checking local model path.")
    #
    # if not os.path.exists(local_model_dir):
    #     print(f"Downloading model to {local_model_dir}...")
    #     snapshot_download(model_id, local_dir=local_model_dir)
    # else:
    #     print(f"Model found at {local_model_dir}, skipping download.")

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

# @asynccontextmanager
# async def lifespan(app: FastAPI) -> AsyncIterator[State]:
#     print("Starting up... checking local model path.")
#
#     if not os.path.exists(local_model_dir):
#         print(f"Downloading model to {local_model_dir}...")
#         snapshot_download(model_id, local_dir=local_model_dir)
#     else:
#         print(f"Model found at {local_model_dir}, skipping download.")
#
#     tokenizer = T5Tokenizer.from_pretrained(local_model_dir, subfolder="tokenizer")
#
#     text_encoder = T5EncoderModel.from_pretrained(
#         local_model_dir, subfolder="text_encoder", torch_dtype=torch_dtype
#     )
#     quantize_(text_encoder, quantization())
#
#     transformer = CogVideoXTransformer3DModel.from_pretrained(
#         local_model_dir, subfolder="transformer", torch_dtype=torch_dtype
#     )
#     quantize_(transformer, quantization())
#
#     vae = AutoencoderKLCogVideoX.from_pretrained(
#         local_model_dir, subfolder="vae", torch_dtype=torch_dtype
#     )
#     quantize_(vae, quantization())
#
#     pipe = CogVideoXPipeline.from_pretrained(
#         local_model_dir,
#         text_encoder=text_encoder,
#         transformer=transformer,
#         vae=vae,
#         torch_dtype=torch_dtype,
#     )
#     pipe.enable_model_cpu_offload()
#     pipe.vae.enable_tiling()
#
#     # app.state.pipeline = pipe
#     print("Pipeline loaded and ready.")
#     yield {"pipe": pipe}
#     print("Shutting down... cleaning up.")
#     del pipe


app = FastAPI(lifespan=lifespan)


# @app.post("/generate")
# def generate(request: Request, prompt: str):
#     pipe = request.state.pipe
#
#     result = pipe(
#         prompt=prompt,
#         height=256,
#         width=256,
#         num_frames=24,
#         num_inference_steps=25,
#         guidance_scale=7.5,
#     ).frames[0]
#
#     output_path = "/tmp/output.mp4"
#     export_to_video(result, output_path, fps=8)
#
#     return {"message": "Video generated", "path": output_path}

@app.post("/generate")
def generate():
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest.\
              The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. \
              Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. \
              Sunlight filters through the tall bamboo, casting a gentle glow on the scene. \
              The panda's face is expressive, showing concentration and joy as it plays. \
              The background includes a small, flowing stream and vibrant green foliage, \
              enhancing the peaceful and magical atmosphere of this unique musical performance."
    video_id = 1

    pipe = app.state.pipeline

    video_id = video_id

    result = pipe(
        prompt=prompt,
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
        # s3_client = boto3.client(
        #     's3',
        #     aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        #     aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        #     region_name=os.environ.get("AWS_REGION", "us-east-1")
        # )

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
