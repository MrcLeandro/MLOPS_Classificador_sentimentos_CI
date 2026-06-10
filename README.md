# Projeto: Classificação de Sentimentos com MLOps
Alunos:
- Márcio Leandro
- Mônica Mendes
- Rudi Modena
  
Obs: Desenvolvido em aula da disciplina Teste de Software juntamente com o prof. Bruno Emílio.

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

## CI/CD Pipeline

Este projeto utiliza uma estratégia de **Continuous Integration** (CI) e **Continuous Deployment** (CD) automático:

### Continuous Integration (CI) com GitHub Actions

O pipeline CI foi configurado no arquivo `.github/workflows/mlops.yml` e executa automaticamente:

1. **Exploração e Limpeza de Dados**
   - Executa o notebook `notebooks/01_exploracao.ipynb`
   - Gera o arquivo `data/tweets_limpo.csv`

2. **Treinamento e Validação do Modelo**
   - Executa o notebook `notebooks/02_pipeline_validacao.ipynb`
   - Produz os artefatos `model.joblib` e `vectorizer.joblib`

3. **Monitoramento de Fairness**
   - Executa o notebook `notebooks/04_monitorar_fairness.ipynb`
   - Avalia o desempenho do modelo por categorias

4. **Testes Automatizados**
   - Executa `pytest test_pipeline.py`
   - Valida a integridade dos artefatos, previsões e fairness

O fluxo de CI é disparado automaticamente a cada push para a branch `main` ou pode ser executado manualmente através do menu "Actions" no GitHub (selecione o workflow "CI Pipeline MLOPs" e clique em "run workflow").

### Continuous Deployment (CD) com Render

A aplicação Streamlit é automaticamente deployada na plataforma **Render** através de CI/CD:

1. **Configuração do Deploy**
   - O arquivo `render.yaml` (ou configuração no dashboard do Render) define como a aplicação deve ser executada
   - A aplicação Streamlit (`app.py`) é servida automaticamente

2. **Fluxo Automático**
   - Cada push para a branch `main` que passa nos testes do CI dispara automaticamente o deployment no Render
   - A aplicação fica disponível em um URL público para acesso em tempo real
   - Não é necessário executar comandos manuais de deploy

3. **Acesso à Aplicação**
   - Após o deploy bem-sucedido, a aplicação está disponível em um URL fornecido pelo Render
   - Usuários podem fazer previsões de sentimento em tempo real sem necessidade de configuração local

### Benefícios da Abordagem CI/CD

- ✅ **Automação completa**: Nenhuma etapa manual necessária após push para `main`
- ✅ **Qualidade garantida**: Testes executados automaticamente antes do deploy
- ✅ **Deploy contínuo**: Mudanças validadas são imediatamente deployadas
- ✅ **Rastreabilidade**: Todo commit é rastreado com seus testes e deploy status
- ✅ **Monitoramento**: Fairness e desempenho monitorados a cada iteração

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
