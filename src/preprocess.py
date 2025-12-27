import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(
    input_path="data/raw/dados_credito_sinteticos.csv",
    output_path="data/processed/dados_credito_processados.csv"
):
    df = pd.read_csv(input_path)

    # 🔹 Separar target
    y = df["aprovado_credito"]

    # 🔹 Remover colunas que NÃO entram no modelo
    X = df.drop(columns=["aprovado_credito", "client_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 Dataset processado (mantendo client_id para uso futuro)
    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed["aprovado_credito"] = y.values
    df_processed["client_id"] = df["client_id"].values

    df_processed.to_csv(output_path, index=False)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
