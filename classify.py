from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm

def classify(logs):
    """
    logs: List of tuples -> [(source, log_message), ...]
    returns: List of labels
    """
    results = []
    for source, log_msg in logs:
        results.append(classify_log(source, log_msg))
    return results



def classify_log(source: str, log_msg: str) -> str:
    """
    Classification Strategy:
    1. LLM → only for legacy / complex systems
    2. Regex → fast, rule-based
    3. BERT → semantic fallback
    """

    # 1️⃣ Legacy systems → LLM (unstructured logs)
    if source.lower().startswith("legacy"):
        return classify_with_llm(log_msg)

    # 2️⃣ Fast path → Regex
    label = classify_with_regex(log_msg)
    if label:
        return label

    # 3️⃣ Intelligent fallback → BERT
    return classify_with_bert(log_msg)
