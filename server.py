import io
import requests
import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException, Body, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from classify import classify
import os 

# ---------------- CONFIG ----------------
CLERK_ISSUER =  os.getenv("CLERK_ISSUER")
JWKS_URL = f"{CLERK_ISSUER}/.well-known/jwks.json"
JWKS = requests.get(JWKS_URL).json()

app = FastAPI(title="Log Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_clerk_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.replace("Bearer ", "")

    try:
        header = jwt.get_unverified_header(token)
        key = next(k for k in JWKS["keys"] if k["kid"] == header["kid"])

        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=CLERK_ISSUER,
            options={"verify_aud": False},
        )
        return payload

    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))



@app.post("/classify/text")
def classify_text(
    log_message: str = Body(..., embed=True),
    source: str = Body("manual"),
    user=Depends(verify_clerk_token),
):
    label = classify([(source, log_message)])[0]
    return {
        "user_id": user["sub"],
        "log_message": log_message,
        "target_label": label,
    }


@app.post("/classify/csv")
async def classify_csv(
    file: UploadFile,
    user=Depends(verify_clerk_token),
):
    if file.content_type != "text/csv":
        raise HTTPException(400, "Only CSV allowed")
    
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV allowed")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    if not {"source", "log_message"}.issubset(df.columns):
        raise HTTPException(400, "CSV must have source & log_message")

    df["target_label"] = classify(
        list(zip(df["source"], df["log_message"]))
    )

    return df.to_dict(orient="records")
