import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from multiprocessing import Pool
from functools import partial

def clean_text(text):
    # Remove special characters and lowercase the text
    # text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    # text = text.lower()
    # remove newlines
    text = text.replace('\n', ' ')
    
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)
    
    # Remove stop words
    # stopwords = nltk.corpus.stopwords.words('english')
    # tokens = [token for token in tokens if token not in stopwords]
    
    # Join the tokens back into a string
    clean_text = ' '.join(tokens)
    
    return clean_text

def process_review(review, analyzer):
    # Extract the review text
    text = review['text']
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Analyze the sentiment of the text
    scores = analyzer.polarity_scores(cleaned_text)
    
    # Add the sentiment scores to the review
    review['sentiment_scores'] = scores

    # Add a sentiment class to the review
    # sentiment compound scores >= 0.05 are positive
    # sentiment compound scores <= -0.05 are negative
    # sentiment compound scores between -0.05 and 0.05 are neutral
    if scores['compound'] >= 0.05:
        review['sentiment_class'] = 'positive'
    elif scores['compound'] <= -0.05:
        review['sentiment_class'] = 'negative'
    else:
        review['sentiment_class'] = 'neutral'
    
    # Remove the review text
    del review['text']
    
    return review

if __name__ == '__main__':

    # Download the necessary NLTK data quietly
    print("Downloading NLTK data...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

    # Initialize the sentiment analyzer
    print("Initializing sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer()

    # Open the filtered_data.json file for reading
    print("Reading in filtered data...")
    with open('filtered_data.json') as f:
        # Load the JSON array from the input file
        reviews = json.load(f)
        print("Analyzing sentiment of {} reviews... (this will most likely take a while" .format(len(reviews)))
        
        # Use a multiprocessing pool to process the reviews in parallel
        with Pool() as pool:
            # Use a partial function to pass the analyzer to the process_review function
            partial_process_review = partial(process_review, analyzer=analyzer)
            processed_reviews = pool.map(partial_process_review, reviews)
        
        # Write the processed reviews to the output file
        with open('sentiment_data.json', 'w') as fout:
            json.dump(processed_reviews, fout)
        
    print("Done!")
