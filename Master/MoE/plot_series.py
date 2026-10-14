import sys
import json

import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = "/home/bruno.costa/Downloads/dataset_global.jsonl"
EVAL_PATH = "evaluation_report.csv"
PRED_PATH = "predictions.csv"
HORIZON = 24
HISTORY = 100


def read_series(path, idx):
    # lê só a linha desejada, sem carregar o dataset inteiro
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return [float(x) for x in json.loads(line)["sequence"]]
    raise IndexError(f"Série {idx} não existe no dataset.")


def main(idx):
    name = f"series_{idx}"

    ev = pd.read_csv(EVAL_PATH)
    row = ev[ev["series"] == name]
    if row.empty:
        raise ValueError(f"{name} não está no evaluation_report.csv")
    row = row.iloc[0]

    actual = [row[f"actual_{i + 1}"] for i in range(HORIZON)]
    pred = [row[f"predicted_{i + 1}"] for i in range(HORIZON)]

    history = read_series(DATASET_PATH, idx)[:-HORIZON]
    n = min(HISTORY, len(history))

    # expert escolhido (opcional)
    expert = ""
    try:
        p = pd.read_csv(PRED_PATH)
        expert = p.loc[p["series"] == name, "selected_experts"].iloc[0]
    except Exception:
        pass

    plt.figure(figsize=(12, 5))
    plt.plot(range(n), history[-n:], label="Histórico")
    plt.plot(range(n, n + HORIZON), actual, "o-", color="black", label="Real")
    plt.plot(range(n, n + HORIZON), pred, "x--", color="tab:red", label="Previsão")
    plt.axvline(n - 0.5, color="gray", linestyle=":")
    plt.title(f"{name} | expert: {expert} | MAE={row['mae']:.4f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"prediction_{name}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main(int(sys.argv[1]))