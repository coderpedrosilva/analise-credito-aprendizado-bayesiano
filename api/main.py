from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.preprocess import preprocess_data
from src.inference import predict_proba

_cache: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _, X_test, _, _, df_test = preprocess_data(return_df=True)
    _cache["X_test"] = X_test
    _cache["df_test"] = df_test
    yield
    _cache.clear()


app = FastAPI(title="API de Crédito Bayesiana", lifespan=lifespan)

app.mount("/ui", StaticFiles(directory="api/static", html=True), name="static")


@app.get("/clientes")
def listar_clientes():
    probs = predict_proba(_cache["X_test"])
    df_test = _cache["df_test"]

    clientes = []
    for i, prob in enumerate(probs):
        clientes.append({
            "cliente": df_test.iloc[i]["client_id"],
            "prob_aprovacao": round(float(prob), 3),
            "status": (
                "Aprovado" if prob >= 0.35
                else "Análise Manual" if prob >= 0.25
                else "Reprovado"
            )
        })

    return clientes
