from transformers import pipeline

print("Loading model... (This may take a while on first run)")
sentiment_pipeline = pipeline("sentiment-analysis")

sentences = [
    "This assignment is very easy and fun!",
    "The weather is terrible today, I don't want to do anything.",
    "Artificial intelligence is the technology of the future."
]

print("\n--- Results ---")
for sentence in sentences:
    result = sentiment_pipeline(sentence)[0]
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})\n")