import requests
from bs4 import BeautifulSoup
import re
import csv
from datetime import datetime

# 1) Get all sitemap URLs from robots.txt
robots_txt = requests.get("https://www.cnbc.com/robots.txt", 
                          headers={"User-Agent": "Mozilla/5.0"}).text

sitemap_urls = []
for line in robots_txt.splitlines():
    if line.startswith("Sitemap:"):
        sitemap_url = line.split(":", 1)[1].strip()
        # only xml sitemaps
        if sitemap_url.endswith(".xml"):
            sitemap_urls.append(sitemap_url)

print(f"Found {len(sitemap_urls)} sitemap files in robots.txt.")

# 2) Prepare regex to match /YYYY/MM/DD/ for years 2019–current
current_year = datetime.now().year
date_pattern = re.compile(r'/((2019|202[0-9])/\d{2}/\d{2})/')

seen = set()

# 3) Open CSV for output
with open("cnbc_articles_2019_present.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["article_url"])

    # 4) Iterate each sitemap
    for sitemap_url in sitemap_urls:
        print(f"Parsing sitemap: {sitemap_url}")
        resp = requests.get(sitemap_url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "xml")

        # If this is a sitemap index, drill into child sitemaps
        if soup.find_all("sitemap"):
            for sm in soup.find_all("sitemap"):
                child_url = sm.loc.text.strip()
                if child_url.endswith(".xml"):
                    sitemap_urls.append(child_url)
            continue

        # Otherwise, it's a URL list
        for loc in soup.find_all("loc"):
            url = loc.text.strip()
            # filter by date pattern
            if date_pattern.search(url):
                if url not in seen:
                    seen.add(url)
                    writer.writerow([url])

print(f"\nDone! Collected {len(seen)} article URLs (2019–present).")
