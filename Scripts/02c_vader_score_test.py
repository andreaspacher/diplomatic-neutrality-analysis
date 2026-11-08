#!/usr/bin/env python3
"""
Interactive VADER sentiment scorer with a 0..1 positivity score.

The raw VADER compound is on -1..+1.
We remap it to 0..1 so that:
    0.0 = most negative
    0.5 = neutral
    1.0 = most positive
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def label(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def to_unit_scale(compound: float) -> float:
    """Remap VADER compound (-1..+1) to 0..1."""
    return (compound + 1.0) / 2.0


def main():
    analyzer = SentimentIntensityAnalyzer()
    print("VADER scorer (0..1 positivity scale) — type a sentence, press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        unit = to_unit_scale(compound)

        print(f"  score (-1..1): {scores['compound']:+.3f}")
        print(f"  compound:     {compound:+.3f}")
        print(f"  pos/neu/neg:  {scores['pos']:.2f} / "
              f"{scores['neu']:.2f} / {scores['neg']:.2f}")
        print(f"  label:        {label(compound)}\n")


if __name__ == "__main__":
    main()