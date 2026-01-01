# ============================================================
# inference.py — Multi-task DistilBERT + SBERT (OFFLINE + GPU)
# Short & Sweet 5–6 line Summary Version
# ============================================================

import torch
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

MODEL_DIR = "model_artifacts"
LOCAL_BERT = "./distilbert-base-uncased"
LOCAL_SBERT = "local_sbert"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------------
# LOAD TOKENIZER
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT)

# -----------------------------
# LOAD ENCODERS
# -----------------------------
le_verdict = joblib.load(f"{MODEL_DIR}/le_verdict.pkl")
le_ipc = joblib.load(f"{MODEL_DIR}/le_ipc.pkl")
mlb_laws = joblib.load(f"{MODEL_DIR}/mlb_laws.pkl")
scaler_pen = joblib.load(f"{MODEL_DIR}/scaler_penalty.pkl")

# -----------------------------
# MODEL (matches training)
# -----------------------------
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

# -----------------------------
# LOAD MODEL WEIGHTS
# -----------------------------
model = MultiTaskModel().to(DEVICE)
state = torch.load(f"{MODEL_DIR}/model_state.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("Model loaded successfully!")

# -----------------------------
# SBERT LOADING
# -----------------------------
sbert = SentenceTransformer(LOCAL_SBERT, device=DEVICE)
embeddings = np.load(f"{MODEL_DIR}/case_embeddings.npy")
case_summaries = np.load(f"{MODEL_DIR}/case_summaries.npy", allow_pickle=True)
nn_model = joblib.load(f"{MODEL_DIR}/nearest_neighbors.joblib")

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(t):
    return " ".join(str(t).lower().replace("\n"," ").replace("\r"," ").split())


# ------------------------------------------------
# SHORT & SWEET 5–6 LINE SUMMARY BUILDER
# ------------------------------------------------
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
            f"Similar past cases show patterns such as: {sim_cases[0]}, {sim_cases[1]}, and {sim_cases[2]}."
        )

    return " ".join(summary)


# -----------------------------
# MAIN PREDICT FUNCTION
# -----------------------------
def predict_case(text):
    cleaned = clean_text(text)

    enc = tokenizer(cleaned, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=256)

    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        pv, pi, pl, pp = model(ids, mask)

    # -------
    # VERDICT
    # -------
    verdict_idx = torch.argmax(pv, 1).cpu().item()
    verdict = le_verdict.inverse_transform([verdict_idx])[0]

    # NOT GUILTY — simple output
    if verdict.lower() == "not guilty":
        summary = build_summary(text, "Not Guilty", None, None, None, None)
        return {
            "verdict": "Not Guilty",
            "ipc_section": None,
            "penalty": None,
            "relevant_laws": None,
            "similar_cases": None,
            "case_summary": summary
        }

    # -------
    # IPC
    # -------
    ipc_idx = torch.argmax(pi, 1).cpu().item()
    ipc_section = le_ipc.inverse_transform([ipc_idx])[0]

    # -------
    # RELEVANT LAWS
    # -------
    law_probs = torch.sigmoid(pl).cpu().numpy()[0]
    idx = np.where(law_probs >= 0.5)[0]
    relevant_laws = mlb_laws.classes_[idx].tolist()
    if len(relevant_laws) == 0:
        relevant_laws = None

    # -------
    # PENALTY
    # -------
    penalty_scaled = float(pp.cpu().item())
    penalty = int(scaler_pen.inverse_transform([[penalty_scaled]])[0][0])

    # -------
    # SBERT SIMILAR CASES
    # -------
    q_emb = sbert.encode([cleaned], convert_to_numpy=True)
    _, sim_idx = nn_model.kneighbors(q_emb, n_neighbors=3)
    sim_cases = [str(s) for s in case_summaries[sim_idx[0]].tolist()]

    # -------
    # SUMMARY
    # -------
    case_summary = build_summary(text, verdict, ipc_section, penalty, relevant_laws, sim_cases)

    return {
        "verdict": verdict,
        "ipc_section": ipc_section,
        "penalty": penalty,
        "relevant_laws": relevant_laws,
        "similar_cases": sim_cases,
        "case_summary": case_summary
    }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    q = "The accused threatened to kill the victim’s family unless ₹50,000 was paid within two days. WhatsApp messages and call recordings verified the threats."
    import json
    print(json.dumps(predict_case(q), indent=4))
