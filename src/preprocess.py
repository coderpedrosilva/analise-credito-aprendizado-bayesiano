import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCALER_PATH = os.path.join("models", "scaler.joblib")


def preprocess_data(
    input_path="data/raw/dados_credito_sinteticos.csv",
    output_path="data/processed/dados_credito_processados.csv",
    return_df=False
):
    df = pd.read_csv(input_path)

    y = df["aprovado_credito"]
    X = df.drop(["aprovado_credito", "client_id"], axis=1)

    # Split antes do fit do scaler para evitar data leakage
    X_train_raw, X_test_raw, y_train, y_test, df_train, df_test = train_test_split(
        X.values,
        y,
        df[["client_id"]].reset_index(drop=True),
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Persiste o scaler para uso em inferência futura
    os.makedirs(os.path.dirname(SCALER_PATH) or ".", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    # Salva dataset processado (treino + teste combinados para referência)
    df_processed = pd.DataFrame(
        X_train, columns=X.columns
    )
    df_processed["aprovado_credito"] = y_train.values
    df_processed["client_id"] = df_train["client_id"].values
    df_processed.to_csv(output_path, index=False)

    if return_df:
        return X_train, X_test, y_train, y_test, df_test

    return X_train, X_test, y_train, y_test
