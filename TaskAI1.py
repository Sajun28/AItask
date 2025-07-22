# Install dependencies first (only once)
# pip install transformers sentencepiece

from transformers import pipeline
# Load summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Input: Long article or paragraph
text = """
Artificial Intelligence (AI) is transforming industries by enabling machines to perform tasks that typically require human intelligence.
These tasks include decision-making, problem-solving, understanding natural language, and visual perception. AI applications range from
autonomous vehicles and healthcare diagnostics to personalized recommendations and smart assistants. The rise of machine learning, a subset
of AI that allows systems to learn from data, has played a major role in recent advances. Deep learning, which uses neural networks with
many layers, is a further evolution of this field and has been particularly successful in areas such as image recognition and language processing.
However, the adoption of AI also brings challenges such as ethical concerns, job displacement, and the need for transparent algorithms.
To fully leverage the benefits of AI, it is essential to develop policies and frameworks that ensure responsible use.
"""

# Generate summary
summary = summarizer(text, max_length=60, min_length=30, do_sample=False)

# Output the summary
print("Summary:\n", summary[0]['summary_text'])
