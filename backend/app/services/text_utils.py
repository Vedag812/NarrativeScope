"""
Text preprocessing utilities for NarrativeScope.

Provides cleaning, normalization, and tokenization functions
used across the search, topic modeling, and network analysis pipelines.
"""

import re
import string
from typing import List


# Common English stop words for political text analysis
POLITICAL_STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
    "in", "with", "to", "for", "of", "not", "no", "can", "had", "have",
    "was", "were", "been", "being", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "this", "that", "these",
    "those", "it", "its", "they", "them", "their", "we", "us", "our",
    "you", "your", "he", "she", "him", "her", "his", "i", "me", "my",
    "am", "are", "if", "then", "than", "so", "just", "about", "also",
    "very", "really", "much", "more", "most", "some", "any", "all",
    "each", "every", "both", "few", "many", "such", "only", "own",
    "same", "other", "new", "old", "first", "last", "long", "great",
    "little", "right", "still", "get", "got", "make", "made", "go",
    "going", "come", "take", "know", "see", "think", "want", "say",
    "said", "like", "well", "back", "even", "here", "there", "when",
    "where", "how", "what", "who", "why", "now", "up", "out", "over",
}


def clean_text(text: str) -> str:
    """
    Clean raw Reddit post text for NLP processing.

    Removes URLs, user mentions, subreddit references, HTML entities,
    and normalizes whitespace while preserving meaningful content.

    Args:
        text: Raw post text from Reddit.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove Reddit-specific markers
    text = re.sub(r'/u/\w+', '', text)  # user mentions
    text = re.sub(r'/r/\w+', '', text)  # subreddit mentions
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)

    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)

    # Remove markdown formatting
    text = re.sub(r'[*_~`#>]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url) -> text

    # Normalize quotes and apostrophes
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize text into lowercase words, optionally removing stop words.

    Args:
        text: Cleaned text string.
        remove_stopwords: Whether to filter out common stop words.

    Returns:
        List of token strings.
    """
    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b[a-z][a-z0-9]{1,}\b', text.lower())

    if remove_stopwords:
        tokens = [t for t in tokens if t not in POLITICAL_STOP_WORDS]

    return tokens


def extract_entities(text: str) -> dict:
    """
    Extract named entities from political text using pattern matching.

    Identifies mentions of political parties, institutions, and key terms
    commonly found in political discourse on Reddit.

    Args:
        text: Cleaned text string.

    Returns:
        Dictionary with entity types as keys and lists of found entities as values.
    """
    entities = {
        "parties": [],
        "institutions": [],
        "topics": [],
    }

    # Political party patterns (US-centric + India)
    party_patterns = [
        (r'\b(?:democrat|democratic|dems?)\b', 'Democratic Party'),
        (r'\b(?:republican|gop|rep)\b', 'Republican Party'),
        (r'\b(?:bjp|bharatiya janata)\b', 'BJP'),
        (r'\b(?:congress|inc)\b', 'Indian National Congress'),
        (r'\b(?:aap|aam aadmi)\b', 'AAP'),
    ]

    # Government institution patterns
    institution_patterns = [
        (r'\b(?:supreme court|scotus)\b', 'Supreme Court'),
        (r'\b(?:congress|parliament|lok sabha|rajya sabha)\b', 'Legislature'),
        (r'\b(?:white house|pmo)\b', 'Executive'),
        (r'\b(?:fbi|cia|ed|cbi)\b', 'Intelligence/Investigation'),
    ]

    text_lower = text.lower()

    for pattern, label in party_patterns:
        if re.search(pattern, text_lower):
            entities["parties"].append(label)

    for pattern, label in institution_patterns:
        if re.search(pattern, text_lower):
            entities["institutions"].append(label)

    return entities


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two text strings.

    Used as a lightweight fallback when embedding-based similarity
    is not available or as an additional ranking signal.

    Args:
        text1: First text string.
        text2: Second text string.

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    tokens1 = set(tokenize(text1))
    tokens2 = set(tokenize(text2))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)
