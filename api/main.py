from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, applications
from pydantic import BaseModel
from service import transcribe

class AsrParam(BaseModel):
    model: str
    compute_type: str | None = None
    language: str | None = None
    align_model: str | None = None
    initial_prompt: Union[str, None] = Query(default=None)
    audio_file: UploadFile = File(...)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/asr/")
async def asr():
    transcribe()
    return None
