# Projeto: Classificação de Sentimentos com MLOps

## Visão geral

Este projeto implementa um fluxo de MLOps para classificação de sentimentos em tweets. Ele inclui:
- Exploração e limpeza de dados
- Construção e validação de pipeline de ML
- Deploy com Streamlit
- Testes automatizados
- Monitoramento de fairness

## Estrutura do projeto

- `app.py` - aplicação Streamlit para previsão de sentimento em texto.
- `data/tweets.csv` - dataset bruto de tweets.
- `data/tweets_limpo.csv` - arquivo de dados limpos gerado pelo notebook de exploração.
- `notebooks/01_exploracao.ipynb` - análise exploratória, limpeza de dados e geração de `tweets_limpo.csv`.
- `notebooks/02_pipeline_validacao.ipynb` - validação, treinamento do modelo e exportação de `model.joblib` e `vectorizer.joblib`.
- `notebooks/03_deploy_streamlit.ipynb` - exemplo de deploy e uso de Streamlit.
- `notebooks/04_monitorar_fairness.ipynb` - análise de fairness e monitoramento de desempenho por tamanho de texto.
- `test_pipeline.py` - testes automatizados para verificação de arquivos, previsões e fairness.
- `.github/workflows/mlops.yml` - pipeline CI que executa notebooks e testes no GitHub Actions.

## Etapas do projeto

1. Exploração de dados
   - Carregar o dataset em `data/tweets.csv`.
   - Realizar análise exploratória e limpeza no notebook `notebooks/01_exploracao.ipynb`.
   - Salvar os dados limpos em `data/tweets_limpo.csv`.

2. Construção da pipeline
   - Treinar e validar modelo no notebook `notebooks/02_pipeline_validacao.ipynb`.
   - Gerar artefatos de inferência: `model.joblib` e `vectorizer.joblib`.

3. Deploy
   - Redeploy local com `app.py` ou usar `notebooks/03_deploy_streamlit.ipynb` como referência.
   - `app.py` carrega `model.joblib` e `vectorizer.joblib` para fazer previsões em tempo real.

4. Testes automatizados
   - `test_pipeline.py` contém casos de teste para:
     - existência dos artefatos de modelo
     - transformações do vectorizer
     - classificação de sentimentos
     - validação dos dados limpos
     - fairness por categorias de tamanho de texto

5. Monitoramento
   - O notebook `notebooks/04_monitorar_fairness.ipynb` avalia a acurácia por grupos de tamanho de texto.
   - Essa etapa ajuda a detectar viés e instabilidade de desempenho.

## Como executar

1. Criar e ativar um ambiente virtual:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Executar notebooks (opcionalmente via GitHub Actions):
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/01_exploracao.ipynb --output 01_exploracao_output.ipynb --output-dir notebooks
   jupyter nbconvert --to notebook --execute notebooks/02_pipeline_validacao.ipynb --output 02_pipeline_validacao_output.ipynb --output-dir notebooks
   jupyter nbconvert --to notebook --execute notebooks/04_monitorar_fairness.ipynb --output 04_monitorar_fairness_output.ipynb --output-dir notebooks
   ```

4. Executar testes:
   ```bash
   pytest test_pipeline.py
   ```

5. Rodar o deploy Streamlit local:
   ```bash
   streamlit run app.py
   ```

## Observações

- O notebook `02_pipeline_validacao.ipynb` gera os arquivos `model.joblib` e `vectorizer.joblib` na raiz do projeto.
- O `app.py` depende desses arquivos para fazer previsões.
- O fluxo de CI do GitHub Actions executa notebooks e testes automaticamente em `main`.
- Foi configurado o disparo manual das ações através do menu "Actions". Selecione o workflow MLOPs Pipeline CI e depois a branch 'main' pelo botão "run workflow".

## Dependências principais

- pandas
- scikit-learn
- great_expectations
- mlflow
- streamlit
- joblib
- matplotlib
- seaborn
- pytest
- jupyter
- nbconvert
