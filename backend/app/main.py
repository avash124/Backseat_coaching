from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import jobs, results, uploads

app = FastAPI(title="Next-Up API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(uploads.router)
app.include_router(jobs.router)
app.include_router(results.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
