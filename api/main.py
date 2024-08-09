from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Form
from pydantic import BaseModel
from service import transcribe
import logging
import time
import random
import string
from typing import Annotated
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()

#fh = logging.handlers.RotatingFileHandler(os.getenv('HOME') + "/whisperx_api_log/api.log", mode="a", maxBytes=10 * 1024 * 1024, backupCount=10)
formatter = logging.Formatter(
    "%(asctime)s - %(thread)d - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)

ch.setFormatter(formatter)
#fh.setFormatter(formatter)
logger.addHandler(ch) #将日志输出至屏幕
#logger.addHandler(fh) #将日志输出至文件

logger = logging.getLogger(__name__)


app = FastAPI(
    title='whisperX service',
    description='whisperX service',
    version='1.0.0',
)


@app.middleware("http")
async def log_requests(request, call_next):
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path} method={request.method}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = '{0:.2f}'.format(process_time)
    logger.info(f"rid={idem} completed_in={formatted_process_time}ms status_code={response.status_code}")

    return response


@app.get("/")
def read_root():
    return {"success": True}


@app.post("/asr")
async def asr(
    model_path: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    compute_type: Annotated[str, Form()] = "float16",   # "float16":GPU; "int8":if low on GPU mem (may reduce accuracy)
    language: Annotated[str, Form()] = None,
    align_model: Annotated[str, Form()] = None,
    initial_prompt: Annotated[str, Form()] = None,
    device: Annotated[str, Form()] = "cuda",    # "cuda":GPU, "cpu":if low on GPU mem
    identify_speaker: Annotated[bool, Form()] = False,
):
    return transcribe(
        model_path,
        compute_type,
        language,
        align_model,
        initial_prompt,
        file.file,
        device,
        identify_speaker,
    )
