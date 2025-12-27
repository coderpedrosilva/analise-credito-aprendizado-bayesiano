from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.preprocess import preprocess_data
from src.inference import predict_proba

app = FastAPI(title="API de Crédito Bayesiana")

# Frontend agora em /ui (não sobrescreve a API)
app.mount("/ui", StaticFiles(directory="api/static", html=True), name="static")

@app.get("/clientes")
def listar_clientes():
    X_train, X_test, y_train, y_test, df_test = preprocess_data(return_df=True)
    probs = predict_proba(X_test)

    clientes = []
    for i, prob in enumerate(probs):
        clientes.append({
            "cliente": df_test.iloc[i]["client_id"],
            "prob_aprovacao": round(float(prob), 3),
            "status": (
                "Aprovado" if prob >= 0.25
                else "Análise Manual" if prob >= 0.1
                else "Reprovado"
            )
        })

    return clientes
