import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", "Colorado Springs, Colorado",
    "Denver, Colorado", "El Cajon, California", "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California", "Oceanside, California",
    "Phoenix, Arizona", "Sacramento, California", "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        for review in reviews:
            review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(reviews, indent=2).encode("utf-8")
            
            # Write your code here
            query = parse_qs(environ["QUERY_STRING"])
            location = query.get("location", [None])[0]
            start_date = query.get("start_date", [None])[0]
            end_date = query.get("end_date", [None])[0]

            filtered_reviews = reviews
            if location and location in VALID_LOCATIONS:
                filtered_reviews = [review for review in filtered_reviews if review["Location"] == location]
            if start_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") >= datetime.strptime(start_date, "%Y-%m-%d")]
            if end_date:
                filtered_reviews = [review for review in filtered_reviews if datetime.strptime(review["Timestamp"], "%Y-%m-%d %H:%M:%S") <= datetime.strptime(end_date, "%Y-%m-%d")]
            
            sorted_reviews = sorted(filtered_reviews, key=lambda x: x["sentiment"]["compound"], reverse=True)

            response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")



            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
                request_body = environ['wsgi.input'].read(request_body_size).decode("utf-8")
                try:
                    review = json.loads(request_body)
                    review["review_id"] = str(uuid.uuid4())
                    review["timestamp"] = datetime.now().isoformat()
                    review["sentiment"] = self.analyze_sentiment(review["review_body"])
                    review["location"] = self.get_location(review["review_body"])
                    review["adj_noun_pairs"] = self.get_adj_noun_pairs(review["review_body"])
                    reviews.append(review)
                    response_body = json.dumps(review, indent=2).encode("utf-8")
                    start_response("201 Created", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]
                except json.JSONDecodeError:
                    #Fallback to parsing as form-encoded
                    review = parse_qs(request_body)
                    review = {k: v[0] for k, v in review.items()}
                    if 'Location' not in review or 'ReviewBody' not in review:
                        start_response("400 Bad Request", [("Content-Type", "application/json")])
                        return [b'{"error": "Missing Location or ReviewBody in request"}']
                    if review['Location'] not in VALID_LOCATIONS:
                        start_response("400 Bad Request", [("Content-Type", "application/json")])
                        return [b'{"error": "Invalid Location"}']
                    review["ReviewId"] = str(uuid.uuid4())
                    review["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])
                    reviews.append(review)

                    response_body = json.dumps(review, indent=2).encode("utf-8")
                    start_response("201 Created", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]
            except Exception as e:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [b'{"error": "Invalid request"}']


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()