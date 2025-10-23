from typing import List, Dict
import os
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_client_accuracy(local_records: List[Dict], out_path: str = "plots/client_accuracy.png"):
    if not local_records:
        return
    ensure_dir(out_path)
    df = pd.DataFrame(local_records)
    clients = sorted(df["client_id"].unique())
    plt.figure(figsize=(10, 6))
    for cid in clients:
        sub = df[df["client_id"] == cid].sort_values("epoch")
        plt.plot(sub["epoch"], sub["acc"], label=f"Client {cid}")
    plt.xlabel("Local Epoch")
    plt.ylabel("Accuracy")
    plt.title("Client Training Accuracy per Epoch")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_student_accuracy(round_eval_records: List[Dict], out_path: str = "plots/student_accuracy_rounds.png"):
    if not round_eval_records:
        return
    ensure_dir(out_path)
    df = pd.DataFrame(round_eval_records)
    df = df.sort_values("round")
    plt.figure(figsize=(8, 5))
    plt.plot(df["round"], df["test_acc"], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title("Student Test Accuracy over Rounds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()