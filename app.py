
import streamlit as st
import joblib
from urllib.parse import quote_plus
from evidence_checker import search_evidence
from news_search import search_news
from nli_evidence_assessor import verify_claim_with_evidence


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load the trained model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "fake_news_pipeline.joblib")

st.title("📰 Fake News & Misinformation Detector")

st.write(
    "Analyze a news article using a machine-learning model "
    "and search for related background information."
)

st.warning(
    "Educational prototype. The model does not independently "
    "verify whether a claim is true or false."
)

# News input
title = st.text_input(
    "News headline",
    key="news_headline"
)

article = st.text_area(
    "News article or description",
    height=200,
    key="news_article"
)

# Analyze news
if st.button("Analyze News", key="analyze_news"):

    content = (title + " " + article).strip()

    if not content:
        st.error("Please enter a headline or article.")

    else:
        # 1. Machine-learning assessment
        probabilities = model.predict_proba([content])[0]
        classes = list(model.classes_)

        fake_probability = float(
            probabilities[classes.index(0)]
        )
        real_probability = float(
            probabilities[classes.index(1)]
        )
        st.session_state["ml_fake_probability"] = fake_probability
        st.session_state["ml_real_probability"] = real_probability

        st.header("1. Model assessment")

        if fake_probability >= 0.60:
            st.warning(
                "The model leans toward the Fake-labelled category."
            )
        elif real_probability >= 0.60:
            st.info(
                "The model leans toward the Real-labelled category."
            )
        else:
            st.info("The model's assessment is uncertain.")

        st.write(
            f"Fake-labelled probability: {fake_probability:.2%}"
        )
        st.progress(fake_probability)

        st.write(
            f"Real-labelled probability: {real_probability:.2%}"
        )
        st.progress(real_probability)

        st.caption(
            "These are model probability estimates, not the "
            "probability that the news is factually false or true."
        )

        # 2. Live background search
        search_query = (
            title.strip()
            if title.strip()
            else " ".join(article.strip().split()[:12])
        )

        st.header("2. Related background information")

        st.write(
            "Searching Wikipedia for potentially relevant pages. "
            "These results are leads for further checking, not "
            "confirmation of the article's claims."
        )

        with st.spinner("Searching for related pages..."):
            results = search_evidence(search_query)

        if results is None:
            st.error(
                "The evidence search failed. Check your internet "
                "connection and try again."
            )

        elif not results:
            st.info("No related Wikipedia pages were found.")

        else:
            for number, item in enumerate(results, start=1):
                with st.expander(f"{number}. {item['title']}"):
                    st.write(item["snippet"])
                    st.markdown(
                        f"[Open source page]({item['url']})"
                    )

        # 3. Additional research links
        encoded_query = quote_plus(search_query)
        nasa_query = quote_plus("site:nasa.gov " + search_query)
        st.header("3. Research further")

        st.link_button(
            "Search Google News",
            f"https://news.google.com/search?q={encoded_query}"
        )

        st.link_button(
            "Search the Web",
            f"https://www.google.com/search?q={encoded_query}"
        )

        st.link_button(
            "Search Fact-Check Reports",
            "https://toolbox.google.com/factcheck/explorer/search/"
            + encoded_query
        )
        st.link_button(
            "Search Official NASA Sources",
            f"https://www.google.com/search?q={nasa_query}"
        )
        st.info(
            "Compare the exact claim, publication date, original "
            "source, and independent reporting. Related search "
            "results alone cannot establish truth."
        )


# 4. Manual evidence assessment
# This section is outside the Analyze News button,
# so it appears before an article is analyzed.

st.divider()

st.header("4. Manual Evidence Assessment")

st.write(
    "Review the sources before deciding what the available "
    "evidence indicates. You must make this assessment yourself; "
    "the machine-learning model does not verify it."
)

evidence_status = st.radio(
    "What does the evidence you reviewed indicate?",
    [
        "Supports the claim",
        "Contradicts the claim",
        "Insufficient evidence"
    ],
    index=2,
    key="manual_evidence_status"
)

evidence_notes = st.text_area(
    "Evidence notes",
    placeholder=(
        "Describe what the source says, which claim it addresses, "
        "and why it supports or contradicts the claim."
    ),
    key="manual_evidence_notes"
)

source_url = st.text_input(
    "Source URL (optional)",
    placeholder="https://example.com/article",
    key="manual_evidence_url"
)

if st.button(
    "Record Evidence Assessment",
    key="record_evidence"
):
    if not evidence_notes.strip():
        st.warning("Please enter your evidence notes first.")

    else:
        st.subheader("Recorded Assessment")

        st.write("**News headline:**", title.strip() or "No headline entered")
        st.info(f"Evidence status (your assessment): {evidence_status}")
        st.write("**Evidence notes:**", evidence_notes)

        if source_url.strip():
            st.write("**Source URL:**", source_url)

        st.caption(
            "This is a user-recorded assessment, not an "
            "independently verified verdict. Check the source's "
            "credibility and whether it addresses the exact claim."
        )
st.divider()
st.header("5. Automated Evidence Assessment (NLI)")

st.write(
    "The system compares your news claim with information "
    "retrieved from Wikipedia and news sources. These results "
    "are automated text comparisons, not proof of truth or falsehood."
)

if st.button("Run Automated Evidence Assessment", key="run_nli_assessment"):

    if not title.strip():
        st.warning("Please enter a news headline first.")

    else:
        with st.spinner("Searching and assessing evidence..."):
            nli_result = verify_claim_with_evidence(title.strip())

        st.session_state["nli_assessment_results"] = nli_result["results"]
        st.session_state["nli_assessed_headline"] = title.strip()

        st.write("**Status:**", nli_result["status"])


if (
    st.session_state.get("nli_assessed_headline") == title.strip()
    and "nli_assessment_results" in st.session_state
):

    nli_results = st.session_state["nli_assessment_results"]

    if not nli_results:
        st.warning("No usable evidence was found.")

    else:
        for item in nli_results:

            with st.expander(item["source_title"]):

                st.write("**Assessment:**", item["assessment"])
                st.write(
                    "**Evidence type:**",
                    item.get("evidence_type", "Not recorded — rerun assessment")
                )
                st.write("**NLI scores:**")

                for label, score in item["scores"].items():
                    st.write(f"{label}: {score:.4f}")

                st.write("**Evidence excerpt:**")
                st.write(item["evidence"][:1000])

                st.write("**Source:**", item["source_url"])

    st.caption(
        "NLI scores measure textual relationships between "
        "the claim and retrieved evidence. They are not "
        "probabilities that the claim is true."
    )
# 5. Live news coverage search

st.divider()
st.header("6. Live News Coverage")

st.write(
    "Compare related reports from different publishers. "
    "Descriptions are provided for context and do not prove "
    "whether a claim is true or false."
)


# Keep the live-news search box synchronized with the main headline
if "previous_news_headline" not in st.session_state:
    st.session_state["previous_news_headline"] = title.strip()
    st.session_state["live_news_query"] = title.strip()

elif st.session_state["previous_news_headline"] != title.strip():
    st.session_state["previous_news_headline"] = title.strip()
    st.session_state["live_news_query"] = title.strip()

news_query = st.text_input(
    "Enter a topic or claim to search for news",
    placeholder="Enter a news topic or claim",
    key="live_news_query"
)

if st.button("Search Live News", key="search_live_news"):

    if not news_query.strip():
        st.warning("Please enter a topic or claim.")

    else:
        with st.spinner("Searching Google News RSS..."):
            news_results = search_news(
                news_query.strip(),
                limit=5
            )

        if news_results is None:
            st.error(
                "News search failed. Check your internet connection "
                "and try again later."
            )

        elif not news_results:
            st.info("No related news articles were found.")

        else:
            for number, item in enumerate(news_results, start=1):
                with st.expander(
                    f"{number}. {item['title']}"
                ):
                    if item.get("description"):
                        st.write("**RSS search snippet:**")
                        st.write(item["description"])
                    else:
                        st.info(
                            "No separate summary is available in this "
                            "Google News RSS result."
                        )
                    if item["url"]:
                        st.markdown(
                            f"[Read the news article]({item['url']})"
                        )

            st.caption(
                "Compare the exact claims and original sources. "
                "Search-result descriptions may be incomplete, "
                "and related coverage alone cannot establish truth."
            )
# 7. Overall assessment

st.divider()
st.header("7. Overall Assessment")

if (
    "ml_fake_probability" in st.session_state
    and st.session_state.get("nli_assessed_headline") == title.strip()
    and "nli_assessment_results" in st.session_state
):

    fake_probability = st.session_state["ml_fake_probability"]
    nli_results = st.session_state["nli_assessment_results"]

    potentially_supported = any(
        item["assessment"] == "Potentially supported"
        for item in nli_results
    )

    possible_contradiction = any(
        item["assessment"] == "Possible contradiction - requires verification"
        for item in nli_results
    )

    if potentially_supported and fake_probability >= 0.60:
        st.warning(
            "Conflicting signals: The ML model leans toward "
            "Fake-labelled, but some retrieved evidence potentially "
            "supports the claim. Further verification is required."
        )

    elif possible_contradiction:
        st.warning(
            "Some retrieved evidence may contradict the claim. "
            "Check the original sources before drawing a conclusion."
        )

    elif potentially_supported:
        st.info(
            "Some retrieved evidence potentially supports the claim. "
            "This is not independent confirmation."
        )

    else:
        st.info(
            "The available evidence is insufficient for an overall "
            "assessment. More reliable sources are needed."
        )

else:
    st.info(
        "Run both the ML analysis and automated evidence assessment "
        "to generate an overall assessment."
    )

st.caption(
    "This overall assessment combines automated signals. "
    "It is not a definitive fact-checking verdict."
)
