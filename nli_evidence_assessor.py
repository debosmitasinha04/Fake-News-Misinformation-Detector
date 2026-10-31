
import onnxruntime as ort
import numpy as np

from tokenizers import Tokenizer
from scipy.special import softmax

from evidence_checker import search_evidence
from news_search import search_news
from article_extractor import extract_article_text

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(BASE_DIR / "models" / "nli_onnx" / "model_quint8_avx2.onnx")
TOKENIZER_PATH = str(BASE_DIR / "models" / "nli_onnx" / "tokenizer.json")


# Load ONNX model
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

# Load tokenizer
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

print("NLI evidence assessor loaded successfully!")


def assess_evidence(claim, evidence):

    # Evidence = premise, Claim = hypothesis
    encoded = tokenizer.encode(
        evidence,
        pair=claim
    )

    input_ids = np.array(
        [encoded.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    outputs = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    probabilities = softmax(outputs[0][0])

    labels = [
        "Contradiction",
        "Entailment",
        "Neutral"
    ]

    results = {
        label: float(probability)
        for label, probability in zip(labels, probabilities)
    }

    return results


def get_assessment(claim, evidence):

    scores = assess_evidence(claim, evidence)

    highest_label = max(scores, key=scores.get)

    if highest_label == "Entailment":
        assessment = "Potentially supported"

    elif highest_label == "Contradiction":
        assessment = "Possible contradiction - requires verification"

    else:
        assessment = "Insufficient evidence"

    return {
        "assessment": assessment,
        "scores": scores,
        "claim": claim,
        "evidence": evidence
    }


def assess_multiple_evidence(claim, evidence_list):

    results = []
    seen_urls = set()

    for item in evidence_list:

        source_url = item.get("url", "")

        if source_url and source_url in seen_urls:
            continue

        if source_url:
            seen_urls.add(source_url)

        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        description = (item.get("description") or "").strip()

        evidence_text = " ".join(
            part for part in [snippet, description]
            if part
        )

        evidence_type = "Headline/description only"

        if source_url:

            extraction_result = extract_article_text(source_url)

            if extraction_result:

                article_text, publisher_url = extraction_result

                if publisher_url:
                    source_url = publisher_url

                if article_text:
                    evidence_text = article_text[:4000]
                    evidence_type = "Full article text"

        if not evidence_text.strip():
            continue

        assessment = get_assessment(
            claim,
            evidence_text
        )

        assessment["source_title"] = item.get("title", "Unknown")
        assessment["source_url"] = source_url
        assessment["evidence_type"] = evidence_type

        results.append(assessment)

    return results


def verify_claim_with_evidence(claim):

    wikipedia_results = search_evidence(claim, limit=3)
    news_results = search_news(claim, limit=3)

    all_evidence = []

    if wikipedia_results:
        all_evidence.extend(wikipedia_results)

    if news_results:
        all_evidence.extend(news_results)

    if not all_evidence:
        return {
            "status": "No evidence found",
            "results": []
        }

    assessments = assess_multiple_evidence(
        claim,
        all_evidence
    )

    return {
        "status": "Evidence assessment completed",
        "results": assessments
    }


if __name__ == "__main__":

    claim = input("\nEnter a news claim: ").strip()

    if claim:

        result = verify_claim_with_evidence(claim)

        print("\nStatus:", result["status"])

        for item in result["results"]:

            print("\nSource:", item["source_title"])
            print("Assessment:", item["assessment"])

            for label, score in item["scores"].items():
                print(f"{label}: {score:.4f}")

            print("URL:", item["source_url"])

    else:
        print("Please enter a claim.")
