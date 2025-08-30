import requests
import csv
import time
from datetime import datetime, timedelta
import calendar

def get_json_with_retry(url, max_retries=3, backoff_factor=2):
    attempts = 0
    while attempts < max_retries:
        try:
            response = requests.get(url, timeout=10)
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            attempts += 1
            wait_time = backoff_factor ** attempts
            print(f"Connection error: {e}. Retrying in {wait_time} seconds (Attempt {attempts}/{max_retries})...")
            time.sleep(wait_time)
    raise Exception("Maximum retries exceeded for URL: " + url)

# Your credentials and query parameters
api_key = 'AIzaSyCuJvBRr4pbL5BO5XbEIDDs33G41kteU40'
cx = 'd07204021650b4db6'
base_query = "US site:cnbc.com"
# Example base query

# Define the overall start and end dates.
start_date = datetime(2021, 9, 1)
end_date = datetime.now()

all_links = []  # To hold all collected article links
current_date = start_date

while current_date < end_date:
    last_day = calendar.monthrange(current_date.year, current_date.month)[1]
    current_end = current_date.replace(day=last_day)
    if current_end > end_date:
        current_end = end_date

    after_date = current_date.strftime("%Y-%m-%d")
    before_date = current_end.strftime("%Y-%m-%d")
    query = f"{base_query} after:{after_date} before:{before_date}"
    print(f"\nProcessing articles from {after_date} to {before_date}...")

    start_index = 1
    while True:
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&start={start_index}"
        print(f"  Requesting results starting at index {start_index}...")
        try:
            response = get_json_with_retry(url, max_retries=5, backoff_factor=2)
        except Exception as e:
            print(f"Error after retries: {e}")
            break

        items = response.get("items", [])
        
        if not items:
            print("  No more items found for this interval.")
            break

        for item in items:
            link = item.get("link")
            if link:
                all_links.append(link)

        next_page = response.get("queries", {}).get("nextPage")
        if next_page:
            start_index = next_page[0].get("startIndex", 0)
        else:
            break

    if current_date.month == 12:
        current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
    else:
        current_date = current_date.replace(month=current_date.month + 1, day=1)

print(f"\nTotal collected article URLs: {len(all_links)}")
for link in all_links:
    print(link)

with open('US_links.csv', 'a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Link"])
    for link in all_links:
        writer.writerow([link])

print("\nThe links have been saved to 'collected_monthly_links.csv'")
