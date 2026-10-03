"""
Dataset Preparation Pipeline
Combines curated conversational & professional sentences with cleaned literature,
produces train/test splits and word frequency dictionary for autocorrect.
"""

import os
import re
import json
import random

# Curated conversational & professional sentences covering diverse real-world contexts
CONVERSATIONAL_SENTENCES = [
    # Greetings & Introductions
    "hello how are you doing today",
    "hi it is great to see you again",
    "good morning everyone welcome to the meeting",
    "good afternoon i hope you are having a productive day",
    "good evening thank you for joining us tonight",
    "it is a pleasure to meet you",
    "nice to meet you i have heard a lot about your work",
    "how has your week been so far",
    "i hope this message finds you well",
    "thank you for reaching out to me",
    
    # Common Questions & Inquiries
    "what are your thoughts on this proposal",
    "can you please send me the updated document",
    "could you clarify what you mean by that",
    "where should we meet for lunch today",
    "what time is the presentation scheduled to begin",
    "how long will it take to finish the task",
    "is there anything else i can assist you with",
    "did you have a chance to review the latest draft",
    "would you like to join us for a quick coffee",
    "what do you think about the new design",
    "can we schedule a quick call tomorrow afternoon",
    "who is responsible for managing this project",
    "why did the system return an unexpected error",
    "how does this algorithm compare to previous methods",
    "when can we expect the final report to be ready",

    # Professional & Business Communication
    "please find attached the project documentation and source code",
    "thank you for your prompt response and helpful feedback",
    "i look forward to hearing from you soon",
    "we are pleased to announce the successful release of our product",
    "let me know if you need any additional information",
    "i will review the pull request and leave my comments",
    "we need to ensure that all requirements are met on time",
    "our team has made significant progress over the last sprint",
    "the meeting has been rescheduled to tomorrow at ten in the morning",
    "please feel free to reach out if you have any questions",
    "we appreciate your continuous support and collaboration",
    "the client was very satisfied with our demonstration",
    "i will follow up with you early next week",
    "let us make sure we align on the key deliverables",
    "as discussed earlier here is the summary of our conversation",

    # AI, Data Science & Machine Learning Contexts
    "machine learning models require high quality training data",
    "artificial intelligence is transforming many industries today",
    "deep learning algorithms are capable of discovering complex patterns",
    "natural language processing allows computers to understand human language",
    "the recurrent neural network predicts the next word in a sentence",
    "long short term memory networks solve the vanishing gradient problem",
    "we evaluated the model using accuracy precision recall and f1 score",
    "the loss function decreased steadily over several training epochs",
    "an n gram language model calculates conditional probability of sequences",
    "perplexity is an important metric for evaluating language models",
    "the autocorrect system suggests the most probable word based on context",
    "data preprocessing involves tokenization normalization and vocabulary building",
    "the embedding layer converts discrete tokens into continuous vectors",
    "we can optimize the hyperparameters using cross validation",
    "the convolutional neural network is commonly used for computer vision",
    "unsupervised learning discovers hidden structures in unlabeled data",
    "reinforcement learning agents learn by interacting with their environment",
    "gradient descent updates the model weights to minimize the loss",
    "the transformer architecture revolutionized modern natural language processing",
    "bidirectional models capture context from both left and right directions",

    # Daily Life, Thoughts & Actions
    "i am going to the library to study for the upcoming exam",
    "it is raining outside so do not forget your umbrella",
    "we decided to take a walk in the park after dinner",
    "the weather is remarkably pleasant and warm this afternoon",
    "i need to buy some fresh groceries on my way home",
    "let us plan a weekend trip with our friends",
    "i really enjoyed reading the new book recommended by the teacher",
    "technology has made communication faster and more accessible than ever",
    "hard work and persistence always lead to meaningful results",
    "it is important to maintain a healthy balance between work and rest",
    "music has a powerful effect on mood and concentration",
    "we celebrated the achievement with our family and close friends",
    "the train was delayed by fifteen minutes due to track maintenance",
    "i will be ready in five minutes and then we can leave",
    "make sure to save your work before closing the application",

    # Expressions of Agreement, Appreciation & Closing
    "i completely agree with your insightful observation",
    "that sounds like an excellent solution to the problem",
    "thank you very much for your kind help and guidance",
    "i truly appreciate your dedication and hard work",
    "have a wonderful day and take good care of yourself",
    "best regards and see you at the conference",
    "it was a great pleasure speaking with you today",
    "we wish you the best of luck in your new endeavors",
    "let us stay in touch and collaborate again in the future",
    "everything went according to plan without any major issues"
]

def clean_gutenberg_text(filepath: str, max_sentences: int = 4000) -> list[str]:
    """Clean and extract sentences from Project Gutenberg text file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Strip header and footer
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    
    start_pos = start_match.end() if start_match else 0
    end_pos = end_match.start() if end_match else len(text)
    
    content = text[start_pos:end_pos]
    
    # Clean whitespace and linebreaks
    content = re.sub(r'\r\n|\r|\n', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    
    # Split into sentences based on punctuation
    raw_sentences = re.split(r'(?<=[.!?])\s+', content)
    cleaned_sentences = []
    
    for s in raw_sentences:
        # Lowercase and clean characters
        s_clean = s.lower()
        # Keep letters, spaces, common apostrophes
        s_clean = re.sub(r"[^a-z\s']", ' ', s_clean)
        # Normalize whitespace
        s_clean = re.sub(r"\s+", ' ', s_clean).strip()
        words = s_clean.split()
        # Keep sentences with reasonable length (4 to 25 words)
        if 4 <= len(words) <= 25:
            cleaned_sentences.append(" ".join(words))
            if len(cleaned_sentences) >= max_sentences:
                break
                
    print(f"Extracted {len(cleaned_sentences)} clean sentences from literature.")
    return cleaned_sentences


def build_full_corpus():
    """Build unified train/test corpus and word frequency dictionary."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    sherlock_path = os.path.join(raw_dir, "sherlock.txt")
    literature_sentences = clean_gutenberg_text(sherlock_path, max_sentences=4500)
    
    # Duplicate conversational sentences to give them high weight in everyday prediction
    conversational_weighted = CONVERSATIONAL_SENTENCES * 15
    
    all_sentences = conversational_weighted + literature_sentences
    random.seed(42)
    random.shuffle(all_sentences)
    
    # Build word frequency table for autocorrect dictionary
    word_freq = {}
    for sentence in all_sentences:
        for word in sentence.split():
            word = word.strip("'")
            if word and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
                
    # Also incorporate Google 10,000 common words if available
    google_words_path = os.path.join(raw_dir, "google_10000_words.txt")
    if os.path.exists(google_words_path):
        with open(google_words_path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if w and w.isalpha() and w not in word_freq:
                    # Baseline prior for valid English vocabulary
                    word_freq[w] = 2
                    
    print(f"Total vocabulary size: {len(word_freq)} unique words")
    
    # Save word frequencies JSON
    freq_path = os.path.join(processed_dir, "word_frequencies.json")
    with open(freq_path, 'w', encoding='utf-8') as f:
        json.dump(word_freq, f, indent=2)
    print(f"Saved word frequencies to {freq_path}")
    
    # Train / Test split (85% train, 15% test)
    split_idx = int(len(all_sentences) * 0.85)
    train_sentences = all_sentences[:split_idx]
    test_sentences = all_sentences[split_idx:]
    
    train_path = os.path.join(processed_dir, "train_corpus.txt")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(train_sentences))
        
    test_path = os.path.join(processed_dir, "test_corpus.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(test_sentences))
        
    full_path = os.path.join(processed_dir, "clean_corpus.txt")
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_sentences))
        
    print(f"Corpus prepared successfully!")
    print(f" - Total sentences: {len(all_sentences)}")
    print(f" - Train sentences: {len(train_sentences)} ({train_path})")
    print(f" - Test sentences:  {len(test_sentences)} ({test_path})")

if __name__ == "__main__":
    build_full_corpus()
