import nltk
import json
from nltk.corpus import wordnet as wn
from collections import defaultdict

def download_nltk_resources():
    """Download required NLTK resources."""
    print("Downloading necessary NLTK resources...")
    nltk.download('wordnet')
    print("Download complete.")

def collect_words_by_pos():
    """Collect words from WordNet and categorize them by part of speech."""
    print("Collecting words from WordNet. This may take a minute...")
    
    # Dictionary to store words by part of speech
    words_by_pos = defaultdict(set)
    
    # WordNet POS mapping
    pos_mapping = {
        'n': 'nouns',
        'v': 'verbs',
        'a': 'adjectives',
        'r': 'adverbs',
        's': 'adjective_satellites'  # Similar to adjectives but serves as complement of the noun
    }
    
    # Get all synsets from WordNet
    for synset in wn.all_synsets():
        # Get the POS and map it to a readable name
        pos = synset.pos()
        pos_name = pos_mapping.get(pos, 'unknown')
        
        # Add lemma names (words) to the appropriate category
        for lemma in synset.lemmas():
            # Only add single-word terms (no phrases with spaces or underscores)
            word = lemma.name()
            if '_' not in word:
                words_by_pos[pos_name].add(word)
    
    # Convert sets to lists for JSON serialization
    result = {pos: sorted(list(words)) for pos, words in words_by_pos.items()}
    
    print(f"Collection complete. Found words in these categories:")
    for pos, words in result.items():
        print(f"  - {pos}: {len(words)} words")
    
    return result

def save_to_json(data, filename="wordnet_words.json"):
    """Save the categorized words to a JSON file."""
    print(f"Saving data to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved successfully to {filename}")

def main():
    """Main function to orchestrate the word collection process."""
    # Ensure required NLTK resources are downloaded
    download_nltk_resources()
    
    # Collect words by part of speech
    words_by_pos = collect_words_by_pos()
    
    # Save the results to a JSON file
    save_to_json(words_by_pos)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
