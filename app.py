# ==========================================================
# app.py — FastAPI Backend (MATCHES inference.py EXACTLY)
# Authentication + Multi-Task DistilBERT + SBERT Similarity
# ==========================================================

import os
import uvicorn
import torch
import joblib
import numpy as np

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

# ---------------- AUTH IMPORT ----------------
from auth import router as auth_router

# ---------------- TRANSFORMERS ----------------
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# ==========================================================
# MODEL CONFIG ( SAME AS inference.py )
# ==========================================================
MODEL_DIR = "model_artifacts"
LOCAL_BERT = "./distilbert-base-uncased"
LOCAL_SBERT = "local_sbert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Using device:", DEVICE)

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT)

# ---------------- LABEL ENCODERS ----------------
le_verdict = joblib.load(f"{MODEL_DIR}/le_verdict.pkl")
le_ipc = joblib.load(f"{MODEL_DIR}/le_ipc.pkl")
mlb_laws = joblib.load(f"{MODEL_DIR}/mlb_laws.pkl")
scaler_pen = joblib.load(f"{MODEL_DIR}/scaler_penalty.pkl")

# ==========================================================
# MODEL DEFINITION (same as training)
# ==========================================================
class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(LOCAL_BERT)
        hidden = self.backbone.config.hidden_size

        self.drop = torch.nn.Dropout(0.2)
        self.head_v = torch.nn.Linear(hidden, len(le_verdict.classes_))
        self.head_i = torch.nn.Linear(hidden, len(le_ipc.classes_))
        self.head_l = torch.nn.Linear(hidden, len(mlb_laws.classes_))
        self.head_p = torch.nn.Linear(hidden, 1)

    def forward(self, ids, mask):
        out = self.backbone(input_ids=ids, attention_mask=mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        return (
            self.head_v(cls),
            self.head_i(cls),
            self.head_l(cls),
            self.head_p(cls).squeeze(-1)
        )


# ---------------- LOAD MODEL WEIGHTS ----------------
model = MultiTaskModel().to(DEVICE)
state = torch.load(f"{MODEL_DIR}/model_state.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("🔥 MODEL LOADED INSIDE FASTAPI!")

# ---------------- LOAD SBERT + NEAREST NEIGHBORS ----------------
sbert = SentenceTransformer(LOCAL_SBERT, device=DEVICE)
embeddings = np.load(f"{MODEL_DIR}/case_embeddings.npy")
case_summaries = np.load(f"{MODEL_DIR}/case_summaries.npy", allow_pickle=True)
nn_model = joblib.load(f"{MODEL_DIR}/nearest_neighbors.joblib")

# ==========================================================
# CLEAN TEXT
# ==========================================================
def clean_text(t):
    return " ".join(str(t).lower().replace("\n", " ").replace("\r", " ").split())


# ==========================================================
# CASE SUMMARY BUILDER (5–6 LINES)
# ==========================================================
def build_summary(facts, verdict, ipc, penalty, laws, sim_cases):
    summary = []

    summary.append(f"This case involves the following facts: {facts}.")
    summary.append(f"The court determined the verdict as {verdict}.")
    if ipc:
        summary.append(f"The primary IPC section applied in the case is {ipc}.")
    if penalty:
        summary.append(f"The penalty imposed is {penalty} months.")
    if laws:
        summary.append(f"Additional relevant legal provisions include: {', '.join(laws)}.")
    if sim_cases:
        summary.append(
            f"Similar past cases show patterns such as: {sim_cases[0]}, "
            f"{sim_cases[1]}, and {sim_cases[2]}."
        )

    return " ".join(summary)


# ==========================================================
# PREDICT FUNCTION USED IN DASHBOARD (MATCHES inference.py)
# ==========================================================
def model_predict(text: str):

    cleaned = clean_text(text)

    enc = tokenizer(
        cleaned,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )

    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pv, pi, pl, pp = model(ids, mask)

    # ---------------- Verdict ----------------
    verdict_idx = torch.argmax(pv, 1).cpu().item()
    verdict = le_verdict.inverse_transform([verdict_idx])[0]

    if verdict.lower() == "not guilty":
        return {
            "verdict": "Not Guilty",
            "ipc_section": None,
            "penalty_months": 0,
            "relevant_laws": [],
            "similar_cases": [],
            "case_summary": build_summary(text, "Not Guilty", None, None, None, None)
        }

    # ---------------- IPC ----------------
    ipc_idx = torch.argmax(pi, 1).cpu().item()
    ipc_section = le_ipc.inverse_transform([ipc_idx])[0]

    # ---------------- Laws ----------------
    law_probs = torch.sigmoid(pl).cpu().numpy()[0]
    idx = np.where(law_probs >= 0.5)[0]
    laws = mlb_laws.classes_[idx].tolist()

    # ---------------- Penalty ----------------
    penalty_scaled = float(pp.cpu().item())
    penalty = float(scaler_pen.inverse_transform([[penalty_scaled]])[0][0])
    penalty = round(penalty, 2)

    # ---------------- Similar Cases (SBERT) ----------------
    q_emb = sbert.encode([cleaned], convert_to_numpy=True)
    _, sim_idx = nn_model.kneighbors(q_emb, n_neighbors=3)
    sim_cases = [str(s) for s in case_summaries[sim_idx[0]].tolist()]

    # ---------------- Summary ----------------
    summary = build_summary(text, verdict, ipc_section, penalty, laws, sim_cases)

    return {
        "verdict": verdict,
        "ipc_section": ipc_section,
        "penalty_months": penalty,
        "relevant_laws": laws,
        "similar_cases": sim_cases,
        "case_summary": summary
    }

# ==========================================================
# FASTAPI SERVER + ROUTES
# ==========================================================

app = FastAPI()

app.include_router(auth_router)

# ---------------- STATIC & TEMPLATES ----------------
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("auth.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ---------------- Prediction API ----------------
class CaseInput(BaseModel):
    case_text: str

@app.post("/predict")
def predict(input_data: CaseInput):
    return model_predict(input_data.case_text)

# ---------------- Run App ----------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
