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
OUTPUT_EVALUATION = "evaluation_report.csv"
OUTPUT_EVALUATION_SUMMARY = "evaluation_summary.txt"
 
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
# MÉTRICAS DE AVALIAÇÃO
# ============================================================
 
def compute_metrics(actual, predicted):
    """
    Calcula métricas de erro entre valores reais e previstos.
    actual e predicted são listas de mesmo tamanho (HORIZON).
    """
 
    errors = [p - a for a, p in zip(actual, predicted)]
    abs_errors = [abs(e) for e in errors]
    squared_errors = [e ** 2 for e in errors]
 
    mae = sum(abs_errors) / len(abs_errors)
    rmse = (sum(squared_errors) / len(squared_errors)) ** 0.5
 
    # MAPE ignorando pontos onde o valor real é ~0 (evita divisão por zero)
    percentage_errors = [
        abs(e) / abs(a)
        for e, a in zip(abs_errors, actual)
        if abs(a) > 1e-8
    ]
 
    mape = (
        (sum(percentage_errors) / len(percentage_errors)) * 100
        if percentage_errors
        else float("nan")
    )
 
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
 
 
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
    evaluation_rows = []
 
 
    # ========================================================
    # PROCESSAR CADA SÉRIE
    # ========================================================
    #
    # IMPORTANTE: o modelo NUNCA recebe os últimos HORIZON
    # pontos de cada série. Esses pontos são guardados como
    # "verdade" (true_future) e usados só depois, para medir
    # o quão boa foi a previsão.
    # ========================================================
 
    for idx, (name, series) in enumerate(
        zip(series_names, series_list)
    ):
 
        # ----------------------------------------------------
        # SPLIT: entrada (visível ao modelo) x holdout (oculto)
        # ----------------------------------------------------
 
        if len(series) <= HORIZON:
            print(
                f"\n[AVISO] {name} tem apenas {len(series)} pontos "
                f"(<= HORIZON={HORIZON}). Não há dados suficientes "
                f"para reservar um holdout de avaliação. Série pulada."
            )
            continue
 
        input_series = series[:-HORIZON]
        true_future = series[-HORIZON:]
 
        print(
            f"\nProcessando {idx + 1}/{len(series_list)} "
            f"- {name} "
            f"(entrada={len(input_series)}, holdout={len(true_future)})"
        )
 
        x = torch.tensor(
            input_series,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)  # (1, T)
 
 
        # ----------------------------------------------------
        # 1. ENCODER (apenas sobre a entrada, sem ver o futuro)
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
        # 2. PREVISÃO + ROUTER (também só a partir da entrada)
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
 
        predicted_values = [float(v) for v in prediction]
 
 
        # ----------------------------------------------------
        # 3. IDENTIFICAR EXPERTS SELECIONADOS
        # ----------------------------------------------------
 
        selected_experts = [
            model.expert_keys[int(expert_idx)]
            for expert_idx in topk_idx
        ]
 
        print(
            "Expert(s) selecionado(s):",
            ", ".join(selected_experts)
        )
 
 
        # ----------------------------------------------------
        # 4. SALVAR PREVISÃO
        # ----------------------------------------------------
 
        prediction_row = {
            "series": name,
            "input_length": len(input_series),
        }
 
        for step, value in enumerate(predicted_values):
 
            prediction_row[
                f"prediction_{step + 1}"
            ] = value
 
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
 
 
        # ----------------------------------------------------
        # 5. COMPARAR PREVISÃO x HOLDOUT REAL
        # ----------------------------------------------------
 
        metrics = compute_metrics(true_future, predicted_values)
 
        print(
            f"Avaliação {name}: "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"MAPE={metrics['mape']:.2f}%"
        )
 
        evaluation_row = {
            "series": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "mape": metrics["mape"],
        }
 
        for step in range(HORIZON):
            evaluation_row[f"actual_{step + 1}"] = float(true_future[step])
            evaluation_row[f"predicted_{step + 1}"] = predicted_values[step]
            evaluation_row[f"error_{step + 1}"] = (
                predicted_values[step] - float(true_future[step])
            )
 
        evaluation_rows.append(evaluation_row)
 
 
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
    # SALVAR RELATÓRIO DE AVALIAÇÃO
    # ========================================================
 
    if not evaluation_rows:
        print(
            "\n[AVISO] Nenhuma série teve pontos suficientes para "
            "avaliação. Relatório de avaliação não foi gerado."
        )
        evaluation_df = pd.DataFrame()
    else:
        evaluation_df = pd.DataFrame(evaluation_rows)
 
        evaluation_df.to_csv(
            OUTPUT_EVALUATION,
            index=False
        )
 
        overall_mae = evaluation_df["mae"].mean()
        overall_rmse = evaluation_df["rmse"].mean()
        overall_mape = evaluation_df["mape"].mean()
 
        best_series = evaluation_df.loc[evaluation_df["mae"].idxmin()]
        worst_series = evaluation_df.loc[evaluation_df["mae"].idxmax()]
 
        summary_lines = [
            "RELATÓRIO DE AVALIAÇÃO - FM-MoE",
            "=" * 50,
            f"Séries avaliadas: {len(evaluation_df)}",
            f"Horizonte de previsão: {HORIZON} passos",
            "",
            "Métricas médias (todas as séries):",
            f"  MAE  médio: {overall_mae:.4f}",
            f"  RMSE médio: {overall_rmse:.4f}",
            f"  MAPE médio: {overall_mape:.2f}%",
            "",
            f"Melhor série: {best_series['series']} "
            f"(MAE={best_series['mae']:.4f})",
            f"Pior série:   {worst_series['series']} "
            f"(MAE={worst_series['mae']:.4f})",
        ]
 
        summary_text = "\n".join(summary_lines)
 
        with open(OUTPUT_EVALUATION_SUMMARY, "w", encoding="utf-8") as f:
            f.write(summary_text)
 
        print("\n" + summary_text)
 
 
    # ========================================================
    # GRÁFICO DA PRIMEIRA SÉRIE AVALIADA (previsto x real)
    # ========================================================
 
    if not evaluation_df.empty:
 
        first_eval = evaluation_df.iloc[0]
 
        first_name = first_eval["series"]
        first_series_full = series_list[
            series_names.index(first_name)
        ]
        first_input = first_series_full[:-HORIZON]
 
        actual_future = [
            first_eval[f"actual_{i + 1}"] for i in range(HORIZON)
        ]
        predicted_future = [
            first_eval[f"predicted_{i + 1}"] for i in range(HORIZON)
        ]
 
        plt.figure(figsize=(12, 5))
 
        history_to_plot = min(100, len(first_input))
        history = first_input[-history_to_plot:]
        history_x = list(range(history_to_plot))
 
        future_x = list(
            range(history_to_plot, history_to_plot + HORIZON)
        )
 
        plt.plot(history_x, history, label="Histórico (entrada)")
 
        plt.plot(
            future_x,
            actual_future,
            marker="o",
            label="Real (holdout)",
            color="black",
        )
 
        plt.plot(
            future_x,
            predicted_future,
            marker="x",
            linestyle="--",
            label="Previsão",
            color="tab:red",
        )
 
        plt.xlabel("Passos temporais")
        plt.ylabel("Valor")
 
        plt.title(
            f"Previsão do FM-MoE vs. Real - {first_name}"
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
 
    if not evaluation_df.empty:
        print(f"Relatório de avaliação salvo em: {OUTPUT_EVALUATION}")
        print(f"Resumo da avaliação salvo em: {OUTPUT_EVALUATION_SUMMARY}")
        print("Gráfico salvo em: prediction_first_series.png")
 
    print("\nDimensão das features:")
    print(features_df.shape)
 
 
if __name__ == "__main__":
    main()
 