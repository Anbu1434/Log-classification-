import io
import os
import requests
import pandas as pd

from fastapi import (
    FastAPI,
    UploadFile,
    HTTPException,
    Body,
    Depends,
    Header,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from classify import classify

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

CLERK_ISSUER = os.getenv("CLERK_ISSUER")

if not CLERK_ISSUER:
    raise RuntimeError("CLERK_ISSUER environment variable not set")

JWKS_URL = f"{CLERK_ISSUER}/.well-known/jwks.json"

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB



app = FastAPI(
    title="Log Classification API",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# MIDDLEWARES
# -------------------------------------------------

@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# -------------------------------------------------
# AUTH HELPERS
# -------------------------------------------------

def get_jwks():
    try:
        response = requests.get(JWKS_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch JWKS")


def verify_clerk_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.replace("Bearer ", "")

    try:
        header = jwt.get_unverified_header(token)
        jwks = get_jwks()

        key = next(
            k for k in jwks["keys"] if k["kid"] == header["kid"]
        )

        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},
        )

        return payload

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Log Classification API is running", "docs_url": "/docs"}


@app.post("/classify/text")
def classify_text(
    log_message: str = Body(..., embed=True),
    source: str = Body("manual"),
    user=Depends(verify_clerk_token),
):
    label = classify([(source, log_message)])[0]

    return {
        "user_id": user.get("sub"),
        "log_message": log_message,
        "target_label": label,
    }


@app.post("/classify/csv")
async def classify_csv(
    file: UploadFile,
    user=Depends(verify_clerk_token),
):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file extension")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    required_columns = {"source", "log_message"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail="CSV must contain 'source' and 'log_message' columns"
        )

    df["target_label"] = classify(
        list(zip(df["source"], df["log_message"]))
    )

    return df.to_dict(orient="records")
