# ============================================================
# train.py — GPU-optimized Multi-Task DistilBERT + SBERT retrieval
# Windows-SAFE version (num_workers=0 + main() wrapper)
#
# Improvements:
# - stronger multi-label learning for relevant_laws (pos_weight + loss multiplier)
# - saves SBERT locally as "local_sbert"
# - prints useful diagnostics for laws
# ============================================================

import os
import random
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ============================================================
# MAIN FUNCTION (required for Windows)
# ============================================================
def main():

    # ------------------------------
    # CONFIG
    # ------------------------------
    # Use the uploaded dataset path
    DATA_FILE = "C:/Users/darsh/OneDrive/Desktop/court_judgment_predictor/balanced_final_dataset_v6.csv"

    OUT_DIR = "model_artifacts"
    os.makedirs(OUT_DIR, exist_ok=True)

    BATCH_SIZE = 8
    EPOCHS = 4
    MAX_LEN = 256
    LR = 3e-5
    WARMUP_PCT = 0.06
    SEED = 42
    MODEL_NAME = "distilbert-base-uncased"

    # LAW loss weight multiplier (increase to force learning)
    LAW_LOSS_WEIGHT = 2.0

    # ------------------------------
    # DEVICE & AMP
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    USE_AMP = True if device.type == "cuda" else False
    print("Mixed precision (AMP) enabled:", USE_AMP)

    # ------------------------------
    # SEED
    # ------------------------------
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    set_seed(SEED)

    # ------------------------------
    # LOAD DATA
    # ------------------------------
    def preprocess(t):
        if not isinstance(t, str):
            return ""
        return " ".join(t.lower().replace("\n"," ").replace("\r"," ").split())

    print("Loading dataset:", DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    print("Rows:", len(df))

    required_cols = [
        "case_fact", "verdict", "ipc_section",
        "relevant_laws", "penalty_months",
        "case_id", "similar_case_summaries"
    ]

    for c in required_cols:
        if c not in df.columns:
            df[c] = ""

    df["case_fact"] = df["case_fact"].astype(str).apply(preprocess)
    df["penalty_months"] = pd.to_numeric(df["penalty_months"], errors="coerce").fillna(0)

    def parse_laws(x):
        if pd.isna(x) or str(x).strip()=="":
            return []
        text = str(x).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                arr = eval(text)
                return [str(i).strip() for i in arr if str(i).strip()]
            except:
                pass
        return [i.strip() for i in text.split(",") if i.strip()]

    df["relevant_laws_list"] = df["relevant_laws"].apply(parse_laws)

    # ------------------------------
    # ENCODERS
    # ------------------------------
    print("Fitting and saving encoders...")
    le_verdict = LabelEncoder()
    le_verdict.fit(df["verdict"].astype(str))
    joblib.dump(le_verdict, f"{OUT_DIR}/le_verdict.pkl")

    le_ipc = LabelEncoder()
    le_ipc.fit(df["ipc_section"].astype(str))
    joblib.dump(le_ipc, f"{OUT_DIR}/le_ipc.pkl")

    mlb_laws = MultiLabelBinarizer()
    mlb_laws.fit(df["relevant_laws_list"])
    joblib.dump(mlb_laws, f"{OUT_DIR}/mlb_laws.pkl")

    scaler_pen = MinMaxScaler()
    df["penalty_scaled"] = scaler_pen.fit_transform(df[["penalty_months"]])
    joblib.dump(scaler_pen, f"{OUT_DIR}/scaler_penalty.pkl")

    print("Encoders saved.")

    # Print some diagnostics for laws
    laws_matrix = mlb_laws.transform(df["relevant_laws_list"])
    pos_counts = laws_matrix.sum(axis=0)
    neg_counts = laws_matrix.shape[0] - pos_counts
    print("Law labels (sample 20 classes):", mlb_laws.classes_[:20])
    print("Law positive counts (sample 20):", pos_counts[:20])

    # compute pos_weight for BCEWithLogitsLoss: pos_weight = neg/pos
    # add epsilon to avoid divide-by-zero for extremely rare labels
    pos_counts = np.array(pos_counts, dtype=np.float32)
    neg_counts = np.array(neg_counts, dtype=np.float32)
    pos_weight = (neg_counts + 1.0) / (pos_counts + 1.0)
    # Convert to torch later (on device)

    # ------------------------------
    # SPLIT
    # ------------------------------
    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=le_verdict.transform(df["verdict"].astype(str))
    )

    print("Train rows:", len(train_df), "Val rows:", len(val_df))

    # ------------------------------
    # TOKENIZER
    # ------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUT_DIR)

    # ------------------------------
    # DATASET
    # ------------------------------
    class CaseDataset(Dataset):
        def __init__(self, df):
            self.text = df["case_fact"].tolist()
            self.verdict = le_verdict.transform(df["verdict"].tolist())
            self.ipc = le_ipc.transform(df["ipc_section"].tolist())
            self.laws = mlb_laws.transform(df["relevant_laws_list"]).astype(np.float32)
            self.penalty = df["penalty_scaled"].astype(np.float32).tolist()

        def __len__(self): return len(self.text)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.text[idx],
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "verdict": torch.tensor(self.verdict[idx], dtype=torch.long),
                "ipc": torch.tensor(self.ipc[idx], dtype=torch.long),
                "laws": torch.tensor(self.laws[idx], dtype=torch.float32),
                "penalty": torch.tensor(self.penalty[idx], dtype=torch.float32)
            }

    # IMPORTANT FIX — num_workers=0 (Windows-safe)
    train_loader = DataLoader(
        CaseDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        CaseDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ------------------------------
    # MODEL
    # ------------------------------
    class MultiTaskModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(MODEL_NAME)
            h = self.backbone.config.hidden_size
            self.drop = nn.Dropout(0.2)

            self.head_v = nn.Linear(h, len(le_verdict.classes_))
            self.head_i = nn.Linear(h, len(le_ipc.classes_))
            self.head_l = nn.Linear(h, len(mlb_laws.classes_))
            self.head_p = nn.Linear(h, 1)

        def forward(self, ids, mask):
            out = self.backbone(input_ids=ids, attention_mask=mask)
            cls = self.drop(out.last_hidden_state[:, 0, :])
            return (
                self.head_v(cls),
                self.head_i(cls),
                self.head_l(cls),
                self.head_p(cls).squeeze(-1)
            )

    model = MultiTaskModel().to(device)

    # ------------------------------
    # OPTIM + SCHED
    # ------------------------------
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_PCT),
        num_training_steps=total_steps
    )

    loss_ce = nn.CrossEntropyLoss()
    # use pos_weight for law BCE
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    loss_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    loss_mse = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_acc = 0.0

    # ------------------------------
    # TRAIN LOOP
    # ------------------------------
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            v = batch["verdict"].to(device)
            i = batch["ipc"].to(device)
            l = batch["laws"].to(device)
            p = batch["penalty"].to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                pv, pi, pl, pp = model(ids, mask)
                loss_v = loss_ce(pv, v)
                loss_i = loss_ce(pi, i)
                loss_l = loss_bce(pl, l)          # BCE with pos_weight
                loss_p = loss_mse(pp, p)

                # Amplify law loss so it gets sufficient gradient
                loss = loss_v + loss_i + LAW_LOSS_WEIGHT * loss_l + loss_p

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/len(train_loader):.4f}"})

        # ------------------------------
        # VALIDATION
        # ------------------------------
        model.eval()
        preds_v, true_v = [], []
        preds_i, true_i = [], []

        all_true_l = []
        all_pred_l = []

        preds_p, true_p = [], []

        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                v = batch["verdict"].cpu().numpy()
                i = batch["ipc"].cpu().numpy()
                l = batch["laws"].cpu().numpy()
                p = batch["penalty"].cpu().numpy()

                pv, pi, pl, pp = model(ids, mask)

                preds_v.extend(torch.argmax(pv, dim=1).cpu().numpy())
                true_v.extend(v)

                preds_i.extend(torch.argmax(pi, dim=1).cpu().numpy())
                true_i.extend(i)

                # multi-label: threshold at 0.25 for validation reporting
                pl_sig = torch.sigmoid(pl).cpu().numpy()
                pred_l_bin = (pl_sig >= 0.25).astype(int)
                all_pred_l.extend(pred_l_bin)
                all_true_l.extend(l)

                preds_p.extend(pp.cpu().numpy())
                true_p.extend(p)

        v_acc = accuracy_score(true_v, preds_v)
        i_acc = accuracy_score(true_i, preds_i)
        try:
            laws_f1 = f1_score(np.array(all_true_l), np.array(all_pred_l), average="macro", zero_division=0)
        except Exception:
            laws_f1 = 0.0
        true_p = np.array(true_p)
        preds_p = np.array(preds_p)
        pen_rmse = float(np.sqrt(np.mean((true_p - preds_p) ** 2)))


        print(f"VAL → VerdictAcc={v_acc:.4f} IPCAcc={i_acc:.4f} LawsF1={laws_f1:.4f} PenRMSE={pen_rmse:.4f}")

        # save best by verdict accuracy
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/model_state.pt")
            tokenizer.save_pretrained(OUT_DIR)
            joblib.dump(le_verdict, os.path.join(OUT_DIR, "le_verdict.pkl"))
            joblib.dump(le_ipc, os.path.join(OUT_DIR, "le_ipc.pkl"))
            joblib.dump(mlb_laws, os.path.join(OUT_DIR, "mlb_laws.pkl"))
            joblib.dump(scaler_pen, os.path.join(OUT_DIR, "scaler_penalty.pkl"))
            print("🔥 Saved best model.")

    # ------------------------------
    # SBERT (retrieval)
    # ------------------------------
    print("Generating SBERT embeddings...")
    # create and save SBERT locally to avoid future downloads
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    emb = sbert.encode(df["case_fact"].tolist(), convert_to_numpy=True, show_progress_bar=True)

    # save local SBERT copy for offline inference
    try:
        sbert.save("local_sbert")
    except Exception as e:
        print("Warning: couldn't save local_sbert:", e)

    np.save(f"{OUT_DIR}/case_embeddings.npy", emb)
    np.save(f"{OUT_DIR}/case_ids.npy", df["case_id"].values)
    np.save(f"{OUT_DIR}/case_summaries.npy", df["similar_case_summaries"].values)

    nn_model = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn_model.fit(emb)
    joblib.dump(nn_model, f"{OUT_DIR}/nearest_neighbors.joblib")

    print("🎉 Training complete!")

# ============================================================
# WINDOWS ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
