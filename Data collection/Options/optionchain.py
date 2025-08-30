import calendar
from datetime import date, timedelta
import requests
import pandas as pd
import time



#
# Your Alpha Vantage API key and stock symbol
APIKEY = 'TJEWCHENCXH1ZU82'  # Replace with your API key
Symbol = 'SPY'          # Replace with your symbol

end_date = date(2020, 7, 31)

# Initialize list and current dateS
weekdays = []
current = end_date - timedelta(days=1)  # start checking from the day before

# Collect the last 35 weekdays
while len(weekdays) < 35:
    if current.weekday() < 5:  # Monday to Friday
        weekdays.append(current)
    current -= timedelta(days=1)


i = 0
while True :
    d = weekdays[i]
    i += 1
    date_str = d.strftime('%Y-%m-%d')
    print(f"Fetching options for {date_str}...")

    url = f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={Symbol}&date={date_str}&apikey={APIKEY}"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            json_data = response.json()

            # Handle API rate limit note
            if "Note" in json_data:
                print(f"Rate limit hit: {json_data['Note']}")
                break  # skip to the next date



            if json_data:
                df = pd.DataFrame(json_data)
                filename = f"{Symbol}'s Chain for {date_str}.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Saved: {filename}")
            else:
                print(f"No options data for {date_str}")

        except Exception as e:
            print(f"JSON parse error for {date_str}: {e}")
            break
    else:
        print(f"Failed request for {date_str} — Status code: {response.status_code}")
        break

    time.sleep(15)  # Stay within API limit (max 4 req/min)
