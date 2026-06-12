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
- Deploy com MLServer
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
- `mlserver/` - diretório com configuração do MLServer para servir o modelo via API REST/gRPC
  - `settings.json` - configurações globais do servidor MLServer
  - `model-settings.json` - configurações específicas do modelo de sentimento
  - `model.py` - implementação da classe `SentimentModel` com lógica de inferência

## Etapas do projeto

1. Exploração de dados
   - Carregar o dataset em `data/tweets.csv`.
   - Realizar análise exploratória e limpeza no notebook `notebooks/01_exploracao.ipynb`.
   - Salvar os dados limpos em `data/tweets_limpo.csv`.

2. Construção da pipeline
   - Treinar e validar modelo no notebook `notebooks/02_pipeline_validacao.ipynb`.
   - Gerar artefatos de inferência: `model.joblib` e `vectorizer.joblib`.

3. Deploy
   - **Deploy local com Streamlit**: Use `app.py` ou `notebooks/03_deploy_streamlit.ipynb` como referência.
     - `app.py` carrega `model.joblib` e `vectorizer.joblib` para fazer previsões em tempo real.
   - **Deploy com MLServer**: Serve o modelo via API REST/gRPC para integração com outras aplicações.
     - Veja a seção [MLServer](#mlserver) para detalhes de configuração e uso.

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

O fluxo de CI é disparado automaticamente a cada push para a branch `main` ou pode ser executado manualmente através do menu "Actions" no GitHub (selecione o workflow "CI Pipeline MLOPs" e clique em "Run workflow").

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

## MLServer

### Visão Geral

[MLServer](https://mlserver.readthedocs.io/) é um servidor de inferência de machine learning que fornece uma interface REST e gRPC para servir modelos de ML. Neste projeto, ele é utilizado para expor o modelo de classificação de sentimentos como um serviço de API.

### Arquivos de Configuração

#### `mlserver/settings.json`
Configurações globais do servidor MLServer:
- **debug**: Ativa modo de debug (true)
- **http_port**: Porta HTTP (8080)
- **grpc_port**: Porta gRPC (8081)
- **metrics_port**: Porta de métricas Prometheus (8082)
- **models**: Lista de modelos disponíveis no servidor

#### `mlserver/model-settings.json`
Configurações específicas do modelo:
- **name**: Nome do modelo (`sentiment-classifier`)
- **implementation**: Classe Python que implementa o modelo (`mlserver.model.SentimentModel`)
- **parameters**: Parâmetros do modelo
  - **uri**: Caminho onde os artefatos do modelo estão armazenados

#### `mlserver/model.py`
Implementação da classe `SentimentModel` que herda de `MLModel`:

**Métodos principais:**
- **`load()`**: Carrega os artefatos `model.joblib` e `vectorizer.joblib` durante a inicialização
- **`predict(payload)`**: Realiza inferência sobre o texto fornecido e retorna:
  - **sentiment**: Classificação (positivo/negativo)
  - **confidence**: Confiança da predição (0 a 1)
  - **probabilities**: Probabilidades para cada classe

### Como Usar o MLServer

#### 1. Instalação
```bash
pip install mlserver mlserver-sklearn
```

#### 2. Executar o Servidor
```bash
mlserver start mlserver/
```

O servidor iniciará nos seguintes endpoints:
- **HTTP**: `http://localhost:8080`
- **gRPC**: `grpc://localhost:8081`
- **Métricas**: `http://localhost:8082/metrics`

#### 3. Fazer Requisições

**Exemplo de requisição HTTP (REST)**:
```bash
curl -X POST \
  http://localhost:8080/v2/models/sentiment-classifier/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [{
      "name": "text",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["Este produto é excelente!"]
    }]
  }'
```

**Resposta esperada**:
```json
{
  "model_name": "sentiment-classifier",
  "model_version": "1.0.0",
  "outputs": [
    {
      "name": "sentiment",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["positivo"]
    },
    {
      "name": "confidence",
      "shape": [1],
      "datatype": "FP32",
      "data": [0.95]
    },
    {
      "name": "probabilities",
      "shape": [2],
      "datatype": "FP32",
      "data": [0.05, 0.95]
    }
  ]
}
```

#### 4. Verificar Status do Modelo
```bash
curl http://localhost:8080/v2/models/sentiment-classifier
```

### Integração com Docker

Para containerizar o MLServer, crie um `Dockerfile`:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080 8081 8082
CMD ["mlserver", "start", "mlserver/"]
```

### Vantagens do MLServer

- ✅ **Padronização**: Segue o padrão V2 Inference Protocol (KServe)
- ✅ **Multi-protocolo**: Suporte a HTTP e gRPC
- ✅ **Observabilidade**: Expõe métricas Prometheus
- ✅ **Escalabilidade**: Integração com Kubernetes e orquestradores
- ✅ **Flexibilidade**: Suporte a múltiplos frameworks (sklearn, XGBoost, TensorFlow, PyTorch, etc.)

## Dependências principais

- pandas
- scikit-learn
- great_expectations
- mlflow
- streamlit
- mlserver
- mlserver-sklearn
- joblib
- matplotlib
- seaborn
- pytest
- jupyter
- nbconvert
