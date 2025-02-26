import json
import nltk
import random
import os
import sys
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None
        self.pos_tags = set()  # Store possible POS tags for this word

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, pos_tag=None):
        """Insert a word into the trie with optional POS tag info"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.word = word
        if pos_tag:
            node.pos_tags.add(pos_tag)
    
    def search(self, word):
        """Search for a word in the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word
    
    def get_words_with_prefix(self, prefix, max_results=10):
        """Get all words with given prefix, up to max_results"""
        result = []
        node = self.root
        
        # Navigate to the prefix node
        for char in prefix:
            if char not in node.children:
                return result
            node = node.children[char]
        
        # Find all words from the prefix node
        self._collect_words(node, result, max_results)
        return result
    
    def _collect_words(self, node, result, max_results):
        """Helper function to collect words from a node and its children"""
        if len(result) >= max_results:
            return
        
        if node.is_end_of_word:
            result.append((node.word, node.pos_tags))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, result, max_results)

class GrammarPredictor:
    def __init__(self, wordlist_path="wordnet_words.json", frequency_data_path="word_frequencies.json", user_data_path="user_preferences.json"):
        # Load word list
        with open(wordlist_path, 'r') as f:
            self.word_dict = json.load(f)
        
        # Download the specific resources needed
        print("Downloading required NLTK resources...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Load or initialize word frequency data
        self.frequency_data = {}
        if os.path.exists(frequency_data_path):
            try:
                with open(frequency_data_path, 'r') as f:
                    self.frequency_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {frequency_data_path} is not valid JSON. Using default frequencies.")
        
        # Load or initialize user preferences with better error handling
        self.user_preferences = defaultdict(Counter)
        self.user_data_path = user_data_path
        self.preference_changes = 0  # Track number of changes since last save
        self.save_threshold = 5      # Save after this many changes
        
        # Ensure the user_data directory exists
        user_data_dir = os.path.dirname(user_data_path)
        if user_data_dir and not os.path.exists(user_data_dir):
            try:
                os.makedirs(user_data_dir)
            except OSError:
                print(f"Warning: Could not create directory {user_data_dir}. User preferences may not be saved.")
        
        # Load existing preferences
        if os.path.exists(user_data_path):
            try:
                with open(user_data_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only try to parse if file is not empty
                        temp_prefs = json.loads(content)
                        for context, words in temp_prefs.items():
                            self.user_preferences[context] = Counter(words)
                print(f"Loaded user preferences from {user_data_path}")
            except json.JSONDecodeError:
                print(f"Warning: {user_data_path} is not valid JSON. Starting with empty preferences.")
            except Exception as e:
                print(f"Error loading user preferences: {str(e)}")
        
        # Context n-grams for semantic matching
        self.context_ngrams = defaultdict(Counter)
        
        # Initialize trie data structure for prefix-based completion
        self.trie = Trie()
        
        # Simple grammar rules for next word prediction
        self.grammar_rules = {
            'DT': ['NN', 'JJ'],  # Determiner -> Noun or Adjective
            'JJ': ['NN'],        # Adjective -> Noun
            'NN': ['VBZ', 'VBP', 'MD', 'IN'],  # Noun -> Verb or Preposition
            'NNS': ['VBP', 'MD', 'IN'],  # Plural Noun -> Verb or Preposition
            'VB': ['DT', 'NN', 'NNS', 'PRP', 'RB'],  # Verb -> Noun or Adverb
            'VBZ': ['DT', 'NN', 'NNS', 'PRP', 'RB'],  # Verb -> Noun or Adverb
            'VBP': ['DT', 'NN', 'NNS', 'PRP', 'RB'],  # Verb -> Noun or Adverb
            'MD': ['VB'],        # Modal -> Verb
            'IN': ['DT', 'PRP', 'NN', 'NNS', 'JJ']  # Preposition -> Noun or Adjective
        }
        
        # Mapping from POS tags to word categories in our dictionary
        self.pos_to_category = {
            'NN': 'nouns',
            'NNS': 'nouns',
            'VB': 'verbs',
            'VBZ': 'verbs',
            'VBP': 'verbs',
            'VBD': 'verbs',
            'VBG': 'verbs',
            'VBN': 'verbs',
            'JJ': 'adjectives',
            'RB': 'adverbs',
            'IN': 'prepositions',
            'DT': 'determiners'
        }
        
        # Reverse mapping: category to POS tags
        self.category_to_pos = {}
        for pos, category in self.pos_to_category.items():
            if category not in self.category_to_pos:
                self.category_to_pos[category] = []
            self.category_to_pos[category].append(pos)
        
        # Now build the trie after setting up category_to_pos
        self._build_trie()
    
    def _build_trie(self):
        """Build the trie data structure from the word dictionary"""
        for category, words in self.word_dict.items():
            pos_tags = self.category_to_pos.get(category, [])
            for word in words:
                for pos in pos_tags:
                    self.trie.insert(word, pos)
    
    def record_selection(self, context, selected_word):
        """Record a user's word selection to improve future suggestions"""
        if not selected_word:
            return
            
        context_key = context.strip().lower()[-20:]  # Use last 20 chars as context key
        self.user_preferences[context_key][selected_word] += 1
        
        # Update the context n-grams for semantic matching
        words = context.split()
        if len(words) > 1:
            prev_word = words[-2] if len(words) >= 2 else ""
            if prev_word:
                self.context_ngrams[prev_word][selected_word] += 1
        
        # Save user preferences periodically
        self.preference_changes += 1
        if self.preference_changes >= self.save_threshold:
            self.save_preferences()
            self.preference_changes = 0
            
    def save_preferences(self):
        """Save user preferences to disk"""
        try:
            with open(self.user_data_path, 'w') as f:
                json.dump(dict(self.user_preferences), f)
            return True
        except Exception as e:
            print(f"Error saving user preferences: {str(e)}")
            return False
    
    def predict_next_pos(self, text):
        """Predict possible part-of-speech categories for the next word"""
        if not text.strip():
            return ['DT', 'NN', 'PRP']  # Start of sentence: likely determiner, noun, or pronoun
        
        # Manual tokenization if NLTK's tokenizer fails
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
        except LookupError:
            # Simple fallback tokenization by spaces and punctuation
            tokens = text.lower().replace('.', ' .').replace(',', ' ,').replace('!', ' !').replace('?', ' ?').split()
            # Default tagging (not accurate but allows the program to run)
            tagged = [(token, 'NN') for token in tokens]  # Default everything to nouns
        
        # Get the POS of the last word
        last_pos = tagged[-1][1]
        
        # Return possible next POS based on grammar rules
        if last_pos in self.grammar_rules:
            return self.grammar_rules[last_pos]
        else:
            # Default to common word types if no specific rule
            return ['NN', 'VB', 'JJ']
    
    def get_prefix_from_text(self, text):
        """Extract the current word being typed (prefix) from text"""
        if not text or text.endswith(' '):
            return ""  # No prefix if text ends with space
        
        # Get the last "word" that's being typed
        words = text.split()
        if not words:
            return ""
        return words[-1]
    
    def suggest_words(self, text, max_suggestions=5):
        """Suggest words based on grammar prediction with ranking and prefix matching"""
        # Get prefix of the current word being typed
        prefix = self.get_prefix_from_text(text)
        
        # Get the text context (excluding current prefix)
        context = text[:-len(prefix)] if prefix else text
        
        # Get possible POS tags for the next word
        possible_pos = self.predict_next_pos(context)
        
        candidates = []
        
        # If we have a prefix, prioritize trie-based completion
        if prefix:
            # Get words matching the prefix from trie
            prefix_matches = self.trie.get_words_with_prefix(prefix, max_results=20)
            
            # Filter by possible POS tags if we have context
            if context:
                filtered_matches = []
                for word, pos_tags in prefix_matches:
                    if any(pos in pos_tags for pos in possible_pos):
                        candidates.append((word, next(iter(pos_tags))))  # Use the first POS tag
                    elif len(filtered_matches) < 5:  # Allow some non-matching POS for variety
                        candidates.append((word, None))
            else:
                # No context, just use all prefix matches
                candidates.extend([(word, next(iter(pos_tags)) if pos_tags else None) for word, pos_tags in prefix_matches])
        
        # If we don't have enough candidates from prefix matching, add more from grammar rules
        if len(candidates) < max_suggestions * 2:
            for pos in possible_pos:
                if pos in self.pos_to_category:
                    category = self.pos_to_category[pos]
                    if category in self.word_dict:
                        # Get words from the appropriate category
                        words = self.word_dict[category]
                        
                        # If we have a prefix, filter by it
                        if prefix:
                            matching_words = [w for w in words if w.startswith(prefix)]
                            sample_size = min(10, len(matching_words))
                            if matching_words:
                                candidates.extend([(word, pos) for word in random.sample(matching_words, sample_size)])
                        else:
                            # No prefix, just sample from category
                            sample_size = min(10, len(words))
                            candidates.extend([(word, pos) for word in random.sample(words, sample_size)])
        
        # Rank candidates based on multiple factors
        ranked_candidates = self.rank_candidates(context, candidates, prefix)
        
        # Return top suggestions
        return [word for word, _ in ranked_candidates[:max_suggestions]]
    
    def rank_candidates(self, context, candidates, prefix=""):
        """Rank word candidates based on multiple factors including prefix matching"""
        context_key = context.strip().lower()[-20:]  # Use last 20 chars as context key
        scored_candidates = []
        
        for word, pos in candidates:
            score = 0
            
            # 1. Word Popularity (0-5 points)
            popularity = self.frequency_data.get(word, 0)
            score += min(popularity * 5, 5)  # Cap at 5 points
            
            # 2. Context Matching (0-3 points)
            if context_key in self.context_ngrams and word in self.context_ngrams[context_key]:
                context_score = min(self.context_ngrams[context_key][word] * 0.5, 3)
                score += context_score
            
            # 3. User Preferences (0-7 points)
            if context_key in self.user_preferences and word in self.user_preferences[context_key]:
                pref_score = min(self.user_preferences[context_key][word] * 1.0, 7)
                score += pref_score
            
            # 4. Prefix exact match bonus (0-4 points)
            if prefix and word.startswith(prefix):
                # Give more points the closer the length is to the prefix
                prefix_ratio = len(prefix) / len(word)
                score += 4 * prefix_ratio
            
            scored_candidates.append((word, score))
        
        # Sort by score (descending)
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    
    def update_frequency_data(self, corpus_file=None):
        """Update word frequency data from a corpus file or use built-in defaults"""
        if corpus_file and os.path.exists(corpus_file):
            # Load external frequency data
            with open(corpus_file, 'r') as f:
                self.frequency_data = json.load(f)
        else:
            # Use some reasonable defaults for common words
            common_words = {
                "the": 1.0, "be": 0.9, "to": 0.9, "of": 0.9, "and": 0.9,
                "a": 0.9, "in": 0.8, "that": 0.8, "have": 0.8, "I": 0.8,
                "it": 0.7, "for": 0.7, "not": 0.7, "on": 0.7, "with": 0.7,
                "he": 0.6, "as": 0.6, "you": 0.6, "do": 0.6, "at": 0.6,
                # More common words can be added
            }
            self.frequency_data.update(common_words)

class SimpleTextUI:
    def __init__(self):
        self.predictor = GrammarPredictor()
        self.predictor.update_frequency_data()
        self.text = ""
        self.predictions = []
        self.feedback_message = ""
        self.feedback_timer = 0
        
    def clear_screen(self):
        """Clear the terminal screen in a cross-platform way"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def display_interface(self):
        """Display the current text and predictions"""
        self.clear_screen()
        print("\n=== Predictive Text Demo ===\n")
        print(f"Text: {self.text}█")  # █ represents cursor
        print("\nPredictions:")
        
        for i, word in enumerate(self.predictions, 1):
            if i <= 4:  # Show only top 4 predictions
                print(f"{i}. {word}")
        
        # Display feedback message if active
        if self.feedback_message and self.feedback_timer > 0:
            print(f"\n{self.feedback_message}")
            self.feedback_timer -= 1
        
        print("\nCommands: [1-4] Select prediction, [Backspace] Delete, [Ctrl+C] Exit")
    
    def run(self):
        """Run the interactive text prediction interface"""
        try:
            while True:
                # Update predictions
                self.predictions = self.predictor.suggest_words(self.text, max_suggestions=4)
                
                # Display interface
                self.display_interface()
                
                # Get user input (single character)
                ch = self.get_char()
                
                # Handle exit
                if ch in ('\x03', '\x04'):  # Ctrl+C or Ctrl+D
                    self.clear_screen()
                    # Save preferences before exiting
                    self.predictor.save_preferences()
                    print("Preferences saved. Goodbye!")
                    break
                
                # Handle backspace
                elif ch in ('\b', '\x7f'):  # Backspace or Delete
                    if self.text:
                        self.text = self.text[:-1]
                
                # Handle number keys for prediction selection
                elif ch in ('1', '2', '3', '4'):
                    prediction_idx = int(ch) - 1
                    if prediction_idx < len(self.predictions):
                        selected_word = self.predictions[prediction_idx]
                        
                        # Get current word being typed
                        current_word = self.predictor.get_prefix_from_text(self.text)
                        
                        # Replace current word with prediction
                        if current_word:
                            self.text = self.text[:-len(current_word)] + selected_word + " "
                        else:
                            self.text += selected_word + " "
                        
                        # Record selection for better future predictions
                        self.predictor.record_selection(self.text, selected_word)
                        
                        # Show feedback
                        self.feedback_message = f"Recorded preference for '{selected_word}'"
                        self.feedback_timer = 3  # Display for 3 interface updates
                
                # Handle regular typing
                elif ch.isprintable():
                    self.text += ch
                
        except KeyboardInterrupt:
            self.clear_screen()
            # Save preferences before exiting
            self.predictor.save_preferences()
            print("Preferences saved. Goodbye!")
    
    def get_char(self):
        """Get a single character from stdin"""
        # Different implementations for different platforms
        try:
            # For Unix-like systems
            import termios
            import tty
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        except ImportError:
            # For Windows
            import msvcrt
            return msvcrt.getch().decode('utf-8')

if __name__ == "__main__":
    ui = SimpleTextUI()
    ui.run()
