from typing import Dict, Any


def preprocess_data(config: Dict[str, Any]) -> Any:
    """Trivial pre-processing stub.

    Returns a constant object; real pipelines would download / clean data here.
    """
    return {"dummy": True}
