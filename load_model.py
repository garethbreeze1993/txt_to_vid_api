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

# print("Starting up... checking local model path.")
#
# if not os.path.exists(local_model_dir):
#     print(f"Downloading model to {local_model_dir}...")
#     snapshot_download(model_id, local_dir=local_model_dir)
# else:
#     print(f"Model found at {local_model_dir}, skipping download.")

# s3_client = boto3.client(
        #     's3',
        #     aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        #     aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        #     region_name=os.environ.get("AWS_REGION", "us-east-1")
        # )