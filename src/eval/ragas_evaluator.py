import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance
from datasets import Dataset
from src.core.config import get_settings

logger = logging.getLogger(__name__)

class RagasEvaluator:
    def __init__(self):
        self.settings = get_settings()
        self.metrics = [faithfulness, answer_relevance]

    def evaluate_response(self, question: str, answer: str, contexts: list[str]) -> dict:
        """
        Runs Ragas evaluation on a single interaction.
        Note: Ragas typically requires an OpenAI API key by default to act as the LLM-as-a-judge.
        In a fully open-source pipeline, you can override the LLM used by Ragas to be a local model.
        For this demo, we'll try to run it. If it fails (due to missing OpenAI key), we'll return a mock score.
        """
        # Ragas expects data in a specific HuggingFace Dataset format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            # Relevancy sometimes requires grounding truth, but we skip it here for simplicity
        }
        dataset = Dataset.from_dict(data)

        try:
            logger.info("Starting Ragas evaluation...")
            # By default, Ragas looks for OPENAI_API_KEY in the environment.
            result = evaluate(dataset, metrics=self.metrics)
            
            # The result is a Ragas Dataset object, we can convert it to pandas or dict
            scores = result.to_pandas().iloc[0].to_dict()
            
            return {
                "faithfulness": scores.get("faithfulness", 0.0),
                "answer_relevance": scores.get("answer_relevance", 0.0)
            }
        except Exception as e:
            logger.warning(f"Ragas evaluation failed (likely missing OpenAI key for judge LLM). Returning mock scores. Error: {e}")
            return {
                "faithfulness": 0.95,
                "answer_relevance": 0.88,
                "mocked": True
            }

evaluator_client = RagasEvaluator()

def get_evaluator() -> RagasEvaluator:
    return evaluator_client
