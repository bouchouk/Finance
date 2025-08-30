import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser
import csv
import time
import os

# CONFIG
INPUT_CSV     = 'articles.csv'
FAILED_CSV    = 'failed_datetimes.csv'
USER_AGENT    = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
PAUSE_SECONDS = 1

# 1) Load your existing CSV

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} links from '{INPUT_CSV}'")

# 2) Add the new column (if it doesn’t already exist)
df['datetime'] = pd.NaT

# 3) Prepare the failed-CSV writer (append mode)
failed_exists = os.path.isfile(FAILED_CSV)
fail_f = open(FAILED_CSV, 'a', newline='', encoding='utf-8')
failed_writer = csv.DictWriter(fail_f, fieldnames=['link'])
if not failed_exists:
    failed_writer.writeheader()

# 4) Loop and fill in the new column, writing failures as they happen
for idx, row in df.iterrows():
    url = row['link']
    print(f"[{idx+1}/{len(df)}] Fetching:", url)
    try:
        resp = requests.get(url, headers={'User-Agent': USER_AGENT})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        time_tag = soup.find('time', attrs={'data-testid': 'published-timestamp'})
        if time_tag is None or 'datetime' not in time_tag.attrs:
            raise ValueError("No <time> tag or missing datetime attr")

        # parse the UTC ISO timestamp
        dt_utc = parser.parse(time_tag['datetime'])

        # store it in the DataFrame
        df.at[idx, 'datetime'] = dt_utc
    
    except Exception as e:
        print("  ✗ Failed:", e)
        failed_writer.writerow({'link': url})
        fail_f.flush()

    time.sleep(PAUSE_SECONDS)

# 5) Close the failures file
fail_f.close()

# 6) Overwrite your input CSV with the new column added
df.to_csv(INPUT_CSV, index=False)
print(f"Done! '{INPUT_CSV}' updated with 'published_datetime' and failures logged to '{FAILED_CSV}'.")
