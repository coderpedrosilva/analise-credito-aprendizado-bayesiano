# 💳 Análise de Aprovação de Crédito com Modelos Bayesianos

Projeto de **análise e previsão de aprovação de crédito** utilizando **Aprendizado Bayesiano**, com geração de dados sintéticos, baseline probabilístico, regressão logística bayesiana, API de inferência e interface web integrada.

---

## 🎯 Objetivo do Projeto

Demonstrar, de forma prática, como **Modelos Bayesianos** podem ser utilizados para:

- Estimar **probabilidades reais de aprovação de crédito**
- Quantificar **incerteza**
- Interpretar estatisticamente o impacto das variáveis
- Disponibilizar previsões via **API REST**
- Visualizar decisões em uma **interface web**

---

## 🧠 Por que Bayes?

- Probabilidades reais ao invés de scores arbitrários  
- Intervalos de credibilidade (HDI)  
- Tomada de decisão baseada em incerteza  
- Padrão utilizado em motores reais de crédito  

---

## 🧪 Modelos Implementados

### 1️⃣ Naive Bayes (Baseline)
- Linha de base probabilística  
- Rápido e interpretável  

### 2️⃣ Regressão Logística Bayesiana (PyMC)
- Inferência MCMC com NUTS  
- Estima distribuições de parâmetros  
- Gera probabilidades calibradas  

---

## 🏗️ Arquitetura

```bash
analise-credito-aprendizado-bayesiano/
├── api/
│   ├── main.py
│   └── static/index.html
├── assets/
├── data/ (gitignored)
├── models/ (gitignored)
│   └── bayesian_credit_trace.nc
├── results/ (gitignored)
├── src/  
│   ├── inference.py
│   ├── generate_data.py
│   ├── pipeline.py
│   └── ...
├── main.py
└── requirements.txt
└── ...
```

---

## ⚙️ Por que Python 3.11?

- Melhor desempenho  
- Melhor gerenciamento de memória  
- Compatibilidade com PyMC, NumPy, sklearn e ArviZ  

---

## 🔄 Pipeline Automatizado

```bash
python main.py
```

O pipeline:

1. Gera dados sintéticos  
2. Pré-processa dados  
3. Treina modelos  
4. Avalia modelos  
5. Persiste modelo bayesiano  
6. Salva métricas e coeficientes  

---

## 🧠 Como acontece o treinamento de Machine Learning

O treinamento neste projeto ocorre de forma **offline e totalmente automatizada**, seguindo um pipeline de engenharia de Machine Learning semelhante ao utilizado em sistemas reais de análise de crédito.

### 1️⃣ Geração dos dados

O processo inicia com a **geração de um dataset sintético realista**, simulando atributos comuns em decisões de crédito, como:

- Idade  
- Renda mensal  
- Score de crédito  
- Valor solicitado  
- Taxa de endividamento  
- Histórico de inadimplência  

Esses dados são gerados por distribuições estatísticas calibradas, permitindo simular cenários reais de concessão de crédito.

### 2️⃣ Construção do rótulo (aprovação)

A variável alvo (`aprovado_credito`) não é aleatória.  
Ela é calculada por uma **função logística de risco**, que combina as variáveis de entrada e gera uma **probabilidade real de aprovação**.

Essa probabilidade é utilizada para gerar o rótulo final de forma estocástica, simulando decisões reais de crédito.

### 3️⃣ Pré-processamento

Antes do treinamento, os dados passam por:

- Normalização (StandardScaler)  
- Separação em treino e teste  
- Organização em matrizes próprias para modelagem  

### 4️⃣ Treinamento dos modelos

Dois modelos são treinados:

- **Naive Bayes** como baseline probabilístico  
- **Regressão Logística Bayesiana** como modelo principal  

O modelo bayesiano é treinado via **MCMC com NUTS (No-U-Turn Sampler)**, estimando distribuições completas dos parâmetros ao invés de apenas valores pontuais.

### 5️⃣ Persistência do modelo

Após o treinamento, o conhecimento aprendido é salvo em disco no formato **NetCDF (`.nc`)**, contendo o posterior bayesiano completo.  
Esse arquivo representa o modelo treinado e é utilizado posteriormente pela API para inferência.

### 6️⃣ Inferência em produção

A API apenas carrega o modelo salvo e utiliza os parâmetros médios das distribuições para calcular **probabilidades individuais de aprovação de crédito**, sem necessidade de retreinamento.

Isso separa claramente:

- Fase de treinamento (offline)  
- Fase de decisão (online)  

---

## 🌐 API de Inferência

Após o treino:

```bash
python -m uvicorn api.main:app --reload
```
---

## 🖥️ Interface Web

Acesse:

```bash
http://127.0.0.1:8000/ui
```

A interface consome a API e exibe:

- Cliente  
- Probabilidade de aprovação  
- Status (Aprovado / Análise Manual / Reprovado)

---

## 🖼️ Demonstração

| Faixa de Probabilidade | Status do Crédito   | Interpretação |
|-----------------------|---------------------|---------------|
| ≥ 0.35                | Aprovado            | Cliente com bom perfil de risco |
| 0.25 – 0.34           | Análise Manual      | Cliente com risco intermediário |
| < 0.25                | Reprovado           | Cliente com alto risco de inadimplência |

![Tela de Análise de Crédito](assets/screenshot-ui.png)

---

## 📈 Interpretação Bayesiana

Coeficientes analisados por HDI 95%.

| Feature | Mean | HDI 2.5% | HDI 97.5% |
|--------|-----|---------|----------|
| coef_3 | -0.486 | -0.839 | -0.130 |
| coef_4 | -0.513 | -0.858 | -0.164 |

Features cujo HDI não cruza zero têm efeito consistente.

---

## 🧩 Conceitos Demonstrados

- Inferência Bayesiana  
- MCMC / NUTS  
- Regressão logística bayesiana  
- Engenharia de pipelines  
- APIs de inferência  
- Visualização de score de crédito  
- Arquitetura de motores de risco  
