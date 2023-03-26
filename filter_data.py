import json
from langdetect import detect
from multiprocessing import Process, Manager
from functools import partial

# Set target params
target_states = ["CA"] # Change to desired states to look for reviews in

# Define paths to input and output files
business_path = "yelp_dataset/yelp_academic_dataset_business.json"
review_path = "yelp_dataset/yelp_academic_dataset_review.json"
output_path = "filtered_data.json"

def is_review_in_english(review):
    try:
        if detect(review['text']) != 'en':
            return False
    except:
        return False
    return True

def is_review_empty(review):
    if len(review['text']) == 0:
        return True
    return False

def is_review_valid(review):
    return not is_review_empty(review) and is_review_in_english(review)

# Define function to filter out businesses not in a specific state
def is_business_in_state(business, state):
    return 'state' in business and business['state'] == state

def process_chunk(chunk, businesses_in_state, prepared_data):
    for review in chunk:
        if review['business_id'] in businesses_in_state and is_review_valid(review):
            prepared_data.append({'text': review['text'], 'stars': review['stars'], 'review_id': review['review_id']})

def main():
    # Define variable to store prepared data
    prepared_data = Manager().list()

    # Read in business data to get list of businesses in set of target states
    print("Reading in business data...")
    businesses_in_state = set()
    with open(business_path, 'r', encoding='utf-8') as f:
        for line in f:
            business = json.loads(line)
            for state in target_states:
                if is_business_in_state(business, state):
                    businesses_in_state.add(business['business_id'])
                    break
    print("Found {} businesses in target states".format(len(businesses_in_state)))

    # Process review data in chunks
    print("Reading in review data... (this may take a while)")
    chunk_size = 10000
    with open(review_path, 'r', encoding='utf-8') as f:
        reviews = []
        for line in f:
            reviews.append(json.loads(line))
            if len(reviews) == chunk_size:
                p = Process(target=process_chunk, args=(reviews, businesses_in_state, prepared_data))
                p.start()
                p.join()
                reviews = []
        if reviews:
            p = Process(target=process_chunk, args=(reviews, businesses_in_state, prepared_data))
            p.start()
            p.join()

    # Convert prepared_data from a shared list to a regular list
    prepared_data = list(prepared_data)

    # Write filtered data to output file
    print("Writing filtered reviews to output file... (this may also take a while)")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prepared_data, f)
    print("Done!")

if __name__ == '__main__':
    main()
