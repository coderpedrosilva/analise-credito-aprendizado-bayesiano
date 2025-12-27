# 💳 Análise de Aprovação de Crédito com Modelos Bayesianos

Projeto end-to-end de **análise e previsão de aprovação de crédito** utilizando **Modelos Bayesianos**, com geração de dados sintéticos, baseline probabilístico, regressão logística bayesiana, interpretação estatística dos coeficientes e automação completa do pipeline em Python.

---

## 🎯 Objetivo do Projeto

Demonstrar, de forma prática e aplicada, como **Modelos Bayesianos** podem ser utilizados para:

- Estimar **probabilidades reais de aprovação de crédito**
- Quantificar **incerteza** nas previsões
- Interpretar estatisticamente o impacto das variáveis
- Comparar abordagens probabilísticas clássicas e bayesianas

Este projeto foi construído com foco em **clareza conceitual**, **reprodutibilidade** e **qualidade de engenharia**.

---

## 🧠 Por que Modelos Bayesianos?

Diferente de modelos puramente frequencistas, a abordagem bayesiana permite:

- Trabalhar explicitamente com **distribuições de probabilidade**
- Incorporar **conhecimento prévio (priors)**
- Obter **intervalos de credibilidade (HDI)** ao invés de apenas estimativas pontuais
- Tomar decisões mais robustas em cenários de risco, como crédito

Isso é especialmente relevante em contextos financeiros, onde **incerteza importa tanto quanto acurácia**.

---

## 🧪 Modelos Implementados

### 1️⃣ Naive Bayes (Baseline)
- Modelo probabilístico clássico
- Serve como **linha de base**
- Rápido, simples e interpretável

### 2️⃣ Regressão Logística Bayesiana (PyMC)
- Modelo bayesiano completo
- Inferência via **NUTS (No-U-Turn Sampler)**
- Estima distribuições para:
  - Intercepto
  - Coeficientes das features
- Permite interpretação estatística profunda dos efeitos

---

## 🏗️ Arquitetura do Projeto

```bash
analise-credito-aprendizado-bayesiano/
│
├── data/ # (ignorado no git)
│ └── processed/ 
│   └── dados_credito_processados.csv # dados pré-processados
│ ├── raw/ 
│   └── dados_credito_sinteticos.csv # dados sintéticos brutos
│
├── results/ # (ignorado no git)
│ ├── bayesian_trace.nc # trace bayesiano (InferenceData)
│ ├── coefficients_summary.csv 
│ └── metrics.json # métricas dos modelos
│
├── src/
│ ├── interpretation/
│ │ └── coefficients.py
│ ├── utils/
│ │ ├── save_results.py
│ │ └── save_trace.py
│ ├── bayesian_logistic.py
│ ├── evaluate_models.py
│ ├── generate_data.py
│ ├── naive_bayes.py
│ ├── pipeline.py
│ └── preprocess.py
│
├── .gitignore
├── main.py
├── README.md 
└── requirements.txt
```

### 🔑 Decisões de Arquitetura
- **Pipeline automatizado** (execução com um único comando)
- Separação clara entre:
  - geração de dados
  - modelagem
  - avaliação
  - persistência de resultados
- Artefatos de dados e resultados **fora do versionamento** (`.gitignore`)
- Estrutura pensada para fácil evolução (novos modelos, novos datasets)

---

## ⚙️ Por que Python 3.11?

- Melhor desempenho geral
- Melhor gerenciamento de memória
- Compatibilidade estável com:
  - NumPy
  - scikit-learn
  - PyMC
  - ArviZ
- Ideal para workloads científicos modernos

---

## 🔄 Pipeline Automatizado

Executar o projeto é simples:

```bash
python main.py
```
O pipeline realiza automaticamente:

1. Criação da estrutura de diretórios (data/, results/)

2. Geração de dataset sintético realista

3. Pré-processamento dos dados

4. Treinamento do modelo Naive Bayes

5. Treinamento do modelo Bayesiano

6. Avaliação dos modelos

7. Salvamento de métricas, trace e coeficientes

8. Interpretação estatística dos resultados

## 📊 Resultados Obtidos
Exemplo de saída ao executar o pipeline:

```text
Copiar código
Naive Bayes
Accuracy: 0.98
ROC AUC: 0.68

Bayesian Logistic Regression
Accuracy: 0.98
ROC AUC: 0.73
```

### 📌 Observação importante:

Apesar de acurácias similares, o modelo bayesiano apresenta:

- Melhor separação probabilística (ROC AUC maior)

- Interpretação estatística robusta

- Medidas explícitas de incerteza

## 📈 Interpretação Bayesiana dos Coeficientes
Os coeficientes do modelo são analisados via intervalos de credibilidade (HDI 95%), permitindo identificar:

- Variáveis com impacto estatisticamente relevante

- Direção do efeito (positivo ou negativo)

- Grau de incerteza associado a cada feature

**Exemplo:** 
| Feature | Mean   | HDI 2.5% | HDI 97.5% |
|--------|--------|----------|-----------|
| coef_3 | -0.486 | -0.839   | -0.130    |
| coef_4 | -0.513 | -0.858   | -0.164    |

➡️ Features cujo HDI não cruza zero possuem efeito consistente na decisão de crédito.

## 🧩 Conceitos Demonstrados

- Inferência Bayesiana

- Regressão Logística Bayesiana

- MCMC e NUTS

- Intervalos de Credibilidade (HDI)

- Avaliação de modelos probabilísticos

- Engenharia de pipelines de ML

- Boas práticas de versionamento
