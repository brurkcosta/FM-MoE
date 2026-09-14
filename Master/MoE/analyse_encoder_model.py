import json
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

from setup.models.modeling_model_encoder import MoERouter


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "moe_model.pt"

DATASET_PATH = "/home/bruno.costa/Downloads/dataset_global.jsonl"

OUTPUT_FEATURES = "encoder_features.csv"
OUTPUT_PREDICTIONS = "predictions.csv"

HORIZON = 24
TOP_K = 1
USE_NOISE = False
DEVICE = "cuda"


# ============================================================
# CARREGAR DATASET
# ============================================================

def load_dataset(path):
    series_list = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            # Cada linha deve ter:
            # {"sequence": [...]}
            if "sequence" not in obj:
                raise ValueError(
                    "Cada linha do JSONL precisa ter a chave 'sequence'."
                )

            sequence = [float(x) for x in obj["sequence"]]

            series_list.append(sequence)

    return series_list


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CARREGANDO MODELO")
    print("=" * 70)

    model = MoERouter.load(
        MODEL_PATH,
        device=DEVICE
    )

    model.eval()

    print(f"Modelo carregado: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Dimensão do embedding: {model.encoder_out_dim}")


    # ========================================================
    # CARREGAR SÉRIES
    # ========================================================

    print("\n" + "=" * 70)
    print("CARREGANDO DATASET")
    print("=" * 70)

    series_list = load_dataset(DATASET_PATH)

    print(f"Quantidade de séries: {len(series_list)}")


    # ========================================================
    # NOMES DAS SÉRIES
    #
    # Como o dataset já foi convertido para "sequence",
    # usamos um nome artificial para cada linha.
    # ========================================================

    series_names = [
        f"series_{i}"
        for i in range(len(series_list))
    ]


    # ========================================================
    # DATAFRAMES DE SAÍDA
    # ========================================================

    feature_rows = []
    prediction_rows = []


    # ========================================================
    # PROCESSAR CADA SÉRIE
    # ========================================================

    for idx, (name, series) in enumerate(
        zip(series_names, series_list)
    ):

        print(
            f"\nProcessando {idx + 1}/{len(series_list)} "
            f"- {name} "
            f"(tamanho={len(series)})"
        )

        x = torch.tensor(
            series,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)  # (1, T)


        # ----------------------------------------------------
        # 1. ENCODER
        # ----------------------------------------------------

        with torch.no_grad():

            embedding = model.encoder(x)

        embedding = embedding.squeeze(0).cpu()

        print(
            f"Embedding gerado: {tuple(embedding.shape)}"
        )


        # ----------------------------------------------------
        # SALVAR FEATURES
        # ----------------------------------------------------

        feature_row = {
            "series": name,
        }

        for feature_idx, value in enumerate(embedding):

            feature_row[
                f"feature_{feature_idx + 1}"
            ] = float(value)

        feature_rows.append(feature_row)


        # ----------------------------------------------------
        # 2. PREVISÃO + ROUTER
        # ----------------------------------------------------

        with torch.no_grad():

            prediction, probs_clean, topk_idx = model(
                x=x,
                horizon=HORIZON,
                dir_csv_experts=Path("analysis_logs"),
                top_k=TOP_K,
                use_noise=USE_NOISE,
                verbose=False,
            )


        prediction = prediction.squeeze(0).cpu()

        probs_clean = probs_clean.squeeze(0).cpu()

        topk_idx = topk_idx.squeeze(0).cpu()


        # ----------------------------------------------------
        # 3. IDENTIFICAR EXPERTS SELECIONADOS
        # ----------------------------------------------------

        selected_experts = []

        for expert_idx in topk_idx:

            expert_name = model.expert_keys[
                int(expert_idx)
            ]

            selected_experts.append(expert_name)


        print(
            "Expert(s) selecionado(s):",
            ", ".join(selected_experts)
        )


        # ----------------------------------------------------
        # 4. SALVAR PREVISÃO
        # ----------------------------------------------------

        prediction_row = {
            "series": name,
            "input_length": len(series),
        }

        for step, value in enumerate(prediction):

            prediction_row[
                f"prediction_{step + 1}"
            ] = float(value)

        for expert_idx, expert_name in enumerate(
            model.expert_keys
        ):

            prediction_row[
                f"weight_{expert_name}"
            ] = float(probs_clean[expert_idx])

        prediction_row["selected_experts"] = ",".join(
            selected_experts
        )

        prediction_rows.append(prediction_row)


    # ========================================================
    # SALVAR FEATURES
    # ========================================================

    features_df = pd.DataFrame(feature_rows)

    features_df.to_csv(
        OUTPUT_FEATURES,
        index=False
    )


    # ========================================================
    # SALVAR PREVISÕES
    # ========================================================

    predictions_df = pd.DataFrame(prediction_rows)

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False
    )


    # ========================================================
    # MOSTRAR PRIMEIRA SÉRIE
    # ========================================================

    first_series = series_list[0]

    # Recarregar a primeira previsão do CSV
    first_prediction = predictions_df.iloc[0]

    predicted_values = [
        first_prediction[f"prediction_{i + 1}"]
        for i in range(HORIZON)
    ]


    # ========================================================
    # GRÁFICO
    # ========================================================

    plt.figure(figsize=(12, 5))

    # Últimos pontos do histórico
    history_to_plot = min(
        100,
        len(first_series)
    )

    history = first_series[-history_to_plot:]

    history_x = list(
        range(history_to_plot)
    )

    future_x = list(
        range(
            history_to_plot,
            history_to_plot + HORIZON
        )
    )

    plt.plot(
        history_x,
        history,
        label="Histórico"
    )

    plt.plot(
        future_x,
        predicted_values,
        marker="o",
        label="Previsão"
    )

    plt.xlabel("Passos temporais")
    plt.ylabel("Valor")

    plt.title(
        "Previsão do FM-MoE - primeira série"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "prediction_first_series.png",
        dpi=200
    )

    plt.show()


    # ========================================================
    # RESUMO
    # ========================================================

    print("\n" + "=" * 70)
    print("ANÁLISE FINALIZADA")
    print("=" * 70)

    print(f"Features salvas em: {OUTPUT_FEATURES}")
    print(f"Previsões salvas em: {OUTPUT_PREDICTIONS}")
    print("Gráfico salvo em: prediction_first_series.png")

    print("\nDimensão das features:")
    print(features_df.shape)

    print("\nPrimeiro embedding:")
    print(
        features_df.iloc[0, 1:].values
    )

    print("\nPrimeira previsão:")
    print(predicted_values)

    print(
        "\nExpert(s) selecionado(s):",
        first_prediction["selected_experts"]
    )


if __name__ == "__main__":
    main()

