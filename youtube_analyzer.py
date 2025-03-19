import re
from typing import Dict, Optional
from pytube import YouTube
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import urllib.request
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class YouTubeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add headers to mimic a browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'youtube\.com\/embed\/([^&\n?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_video_details(self, url: str) -> Dict:
        """Fetch video details using pytube with custom headers."""
        try:
            # Create a custom opener with headers
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', self.headers['User-Agent'])]
            urllib.request.install_opener(opener)

            # Initialize YouTube object with custom headers
            yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
            
            # Get video details
            return {
                'title': yt.title,
                'channel': yt.author,
                'description': yt.description,
                'views': yt.views,
                'length': yt.length,
                'publish_date': yt.publish_date
            }
        except Exception as e:
            raise ValueError(f"Error fetching video details: {str(e)}")

    def analyze_content(self, text: str) -> Dict:
        """Analyze the content using NLP techniques."""
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentiment analysis
        sentiment = blob.sentiment
        
        # Get key phrases (noun phrases)
        key_phrases = blob.noun_phrases
        
        # Get word frequency
        words = [word.lower() for word in blob.words if word.lower() not in self.stop_words]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'sentiment': sentiment,
            'key_phrases': list(key_phrases),
            'top_words': sorted_words[:10],
            'summary': self.generate_summary(text)
        }

    def generate_summary(self, text: str, num_sentences: int = 3) -> str:
        """Generate a summary of the text."""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on word frequency
        word_freq = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                if word not in self.stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        sentence_scores = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                if word in word_freq:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]
        
        # Get top sentences
        summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        summary_sentences = [sentence for sentence, _ in sorted(summary_sentences, key=lambda x: sentences.index(x[0]))]
        
        return ' '.join(summary_sentences)

    def analyze_video(self, video_url: str) -> Dict:
        """Main method to analyze a YouTube video."""
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        # Get video details
        video_details = self.get_video_details(video_url)
        
        # Combine title and description for analysis
        content = f"{video_details['title']} {video_details['description']}"
        
        # Analyze content
        analysis = self.analyze_content(content)
        
        return {
            'title': video_details['title'],
            'channel': video_details['channel'],
            'views': video_details['views'],
            'length': video_details['length'],
            'publish_date': video_details['publish_date'],
            'analysis': analysis
        }

def main():
    try:
        analyzer = YouTubeAnalyzer()
        video_url = input("Enter YouTube video URL: ")
        
        print("\nAnalyzing video...")
        results = analyzer.analyze_video(video_url)
        
        print("\n=== Video Analysis Results ===")
        print(f"\nTitle: {results['title']}")
        print(f"Channel: {results['channel']}")
        print(f"Views: {results['views']:,}")
        print(f"Length: {results['length']} seconds")
        print(f"Published: {results['publish_date']}")
        
        print("\nSummary:")
        print(results['analysis']['summary'])
        
        print("\nKey Topics:")
        for phrase in results['analysis']['key_phrases'][:5]:
            print(f"- {phrase}")
        
        print("\nSentiment Analysis:")
        print(f"Polarity: {results['analysis']['sentiment'].polarity:.2f}")
        print(f"Subjectivity: {results['analysis']['sentiment'].subjectivity:.2f}")
        
        print("\nTop Keywords:")
        for word, freq in results['analysis']['top_words']:
            print(f"- {word}: {freq}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the video URL is correct and the video is public")
        print("2. Check your internet connection")
        print("3. Try a different video URL")
        print("4. If the issue persists, try updating pytube: pip install --upgrade pytube")

if __name__ == "__main__":
    main() 