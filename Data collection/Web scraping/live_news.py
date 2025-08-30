import feedparser
from datetime import datetime
import csv

# Define the RSS feed URLs for the five websites.
rss_feeds = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "NYTimes": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "BBC": "http://feeds.bbci.co.uk/news/business/rss.xml",
    "YahooFinance": "https://finance.yahoo.com/rss/headline",
    "Investopedia": "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headlines"
}

# Set the start date (January 1, 2019) and end date (today).
start_date = datetime(2019, 1, 1)
end_date = datetime.now()

# This list will store tuples of (source, publication_date, link)
collected_articles = []

def parse_date(entry):
    # Try to extract a published date from common attributes.
    pub_date = None
    if 'published_parsed' in entry and entry.published_parsed:
        pub_date = datetime(*entry.published_parsed[:6])
    elif 'updated_parsed' in entry and entry.updated_parsed:
        pub_date = datetime(*entry.updated_parsed[:6])
    return pub_date

# Process each RSS feed.
for source, feed_url in rss_feeds.items():
    print(f"Processing feed from {source}...")
    feed = feedparser.parse(feed_url)
    
    if feed.bozo:
        print(f"Error parsing {source} feed: {feed.bozo_exception}")
        continue

    for entry in feed.entries:
        pub_date = parse_date(entry)
        if not pub_date:
            continue  # Skip if no date is available.
        if start_date <= pub_date <= end_date:
            # Append a tuple (source, publication_date, link)
            collected_articles.append((source, pub_date.strftime("%Y-%m-%d"), entry.link))

print(f"\nCollected {len(collected_articles)} articles from {start_date.date()} to {end_date.date()}.\n")

# Sort articles by publication date (oldest first) if desired.
collected_articles.sort(key=lambda x: x[1])

# Print the list of collected articles.
for source, pub_date, link in collected_articles:
    print(f"[{source}] {pub_date} - {link}")

# (Optional) Save the list to a CSV file.
with open('financial_news_links.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Source", "Publication Date", "Link"])
    for article in collected_articles:
        writer.writerow(article)

print("\nThe article links have been saved to 'financial_news_links.csv'")
