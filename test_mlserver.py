import pytest
import requests
import time

MLSERVER_URL = "http://localhost:8080"
MODEL_NAME = "sentiment-classifier"
HEALTH_ENDPOINT = f"{MLSERVER_URL}/health"
PREDICT_ENDPOINT = f"{MLSERVER_URL}/v2/models/{MODEL_NAME}/infer"

class TestMLServerIntegration:
    @pytest.fixture(scope="session", autouse=True)
    def wait_for_server(self):
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(HEALTH_ENDPOINT, timeout=5)
                if response.status_code == 200:
                    print("✓ MLServer está pronto")
                    break
            except requests.exceptions.ConnectionError:
                if i < max_retries - 1:
                    print(f"Aguardando MLServer... ({i+1}/{max_retries})")
                    time.sleep(1)
                else:
                    raise RuntimeError("MLServer não iniciou após 30 segundos")

    def test_health_check(self):
        response = requests.get(HEALTH_ENDPOINT)
        assert response.status_code == 200

    def test_model_metadata(self):
        url = f"{MLSERVER_URL}/v2/models/{MODEL_NAME}"
        response = requests.get(url)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == MODEL_NAME

    def test_positive_sentiment_prediction(self):
        payload = {"inputs": [{"name": "text", "shape": [1], "datatype": "BYTES", "data": ["Este produto é excelente!"]}]}
        response = requests.post(PREDICT_ENDPOINT, json=payload)
        assert response.status_code == 200
        data = response.json()
        sentiment = data["outputs"][0]["data"][0]
        assert sentiment in ["positivo", "negativo"]

    def test_prediction_response_structure(self):
        payload = {"inputs": [{"name": "text", "shape": [1], "datatype": "BYTES", "data": ["Teste"]}]}
        response = requests.post(PREDICT_ENDPOINT, json=payload)
        data = response.json()
        assert "model_name" in data
        assert data["model_name"] == MODEL_NAME
        assert "outputs" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
