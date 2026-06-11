import os
import joblib
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.utils import get_model_uri

class SentimentModel(MLModel):
    async def load(self) -> bool:
        model_uri = get_model_uri(self._settings)
        model_dir = model_uri or "."
        model_path = os.path.join(model_dir, "model.joblib")
        vectorizer_path = os.path.join(model_dir, "vectorizer.joblib")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Modelo ou vetorizador não encontrado em: {model_dir}")
        
        self._model = joblib.load(model_path)
        self._vectorizer = joblib.load(vectorizer_path)
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        text = payload.inputs[0].data[0]
        if not isinstance(text, str):
            text = str(text)
        
        text_vectorized = self._vectorizer.transform([text])
        prediction = self._model.predict(text_vectorized)[0]
        probabilities = self._model.predict_proba(text_vectorized)[0]
        sentiment_label = "positivo" if prediction == 1 else "negativo"
        confidence = float(max(probabilities))
        
        return InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(name="sentiment", shape=[1], datatype="BYTES", data=[sentiment_label]),
                ResponseOutput(name="confidence", shape=[1], datatype="FP32", data=[confidence]),
                ResponseOutput(name="probabilities", shape=[2], datatype="FP32", data=probabilities.tolist()),
            ],
        )
