from fastapi import FastAPI
import pandas as pd

from src.preprocess import preprocess_data
from src.bayesian_logistic import train_bayesian_logistic
from src.predict import predict_probabilities

app = FastAPI(title="API de Análise de Crédito Bayesiana")

@app.get("/clientes")
def listar_clientes():
    X_train, X_test, y_train, y_test, df_test = preprocess_data(return_df=True)

    model, trace = train_bayesian_logistic(X_train, y_train)
    probs = predict_probabilities(model, trace, X_test)

    clientes = []
    for i, prob in enumerate(probs):
        clientes.append({
            "cliente": df_test.iloc[i]["client_id"],
            "prob_aprovacao": round(float(prob), 3),
            "status": (
                "Aprovado" if prob >= 0.75
                else "Análise Manual" if prob >= 0.6
                else "Reprovado"
            )
        })

    return clientes
