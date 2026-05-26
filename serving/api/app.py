"""FastAPI translation server for Hindi Jinnie."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from translator import TranslatorService

translator: TranslatorService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global translator
    translator = TranslatorService()
    yield


app = FastAPI(title="Hindi Jinnie API", version="2.0.0", lifespan=lifespan)

origins = os.environ.get(
    "CORS_ORIGINS",
    "https://projects.tarun-ssharma.com,http://localhost:8888,http://127.0.0.1:8888",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["fast", "accurate"] = "fast"


class TranslateResponse(BaseModel):
    translation: str
    latency_ms: int
    mode: str


@app.get("/health")
def health():
    if translator is None:
        return {"status": "starting"}
    info = translator.health()
    return {"status": "ok", **info}


@app.post("/translate", response_model=TranslateResponse)
def translate(body: TranslateRequest):
    if translator is None:
        raise HTTPException(503, "Model not loaded")
    try:
        result = translator.translate(body.text, mode=body.mode)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return TranslateResponse(**result)
