import newspaper
import json
import pandas as pd
import csv
import time
import requests
# df = pd.read_csv('google_finance_news_links')


# url ='https://www.investopedia.com/s-and-p-500-gains-and-losses-today-chip-stocks-drop-as-top-firms-cite-export-restriction-impacts-11716729'
  
# article = newspaper.Article(url=url, language='en')
# article.download()
# article.parse()

# article ={
#     "title": str(article.title),
#     "text": str(article.text),
#     "authors": article.authors,
#     "published_date": str(article.publish_date),
#     "top_image": str(article.top_image),
#     "videos": article.movies,
#     "keywords": article.keywords,
#     "summary": str(article.summary)
# }


# print(article["text"])


# read your links
# df = pd.read_csv('cnbc_articles_2019_present.csv')
# print(f"Read {len(df)} links from CSV.")

# articles_data = []

# count = 0

# for index, row in df.iterrows():
#     if count == 5000 :
#         break
#     count += 1
#     url = row['Link']
#     print(f"Processing ({index+1}/{len(df)}): {url}")
#     try:
#         a = newspaper.Article(url, language='en')
#         a.download(); a.parse()
        
#         article_data = {
#             "link":            url,
#             "title":           a.title,
#             "text":            a.text,
#             "published_date":  a.publish_date.isoformat() if a.publish_date else "",
#             "keywords":        "; ".join(a.keywords),
#         }
#         articles_data.append(article_data)
#     except Exception as e:
#         print(f"  ✗ Failed: {e}")
#     time.sleep(1)

# # Build a DataFrame straight from the list of dicts
# df_articles = pd.DataFrame(articles_data)

# # If you want only those 5 columns in that order:
# df_articles = df_articles[["link", "title", "text", "published_date", "keywords"]]

# # Save out
# df_articles.to_csv('articles.csv', index=False, encoding='utf-8')
# print(f"Saved {len(df_articles)} articles to 'articles.csv'")


import newspaper
import pandas as pd
import csv
import time
import os

# read your links
df = pd.read_csv('cnbc_articles_2019_present.csv')
print(f"Read {len(df)} links from CSV.")

# output files and their headers
output_file       = 'articles.csv'
failed_file       = 'failed_links.csv'
fieldnames        = ["link", "title", "text", "published_date", "keywords"]
failed_fieldnames = ["link"]

# check if each file already exists (so we know whether to write headers)
file_exists   = os.path.isfile(output_file)
failed_exists = os.path.isfile(failed_file)

# open both files in append mode
with open(output_file, 'a', newline='', encoding='utf-8') as out_csv, \
     open(failed_file, 'a', newline='', encoding='utf-8') as fail_csv:

    writer        = csv.DictWriter(out_csv,    fieldnames=fieldnames)
    failed_writer = csv.DictWriter(fail_csv,    fieldnames=failed_fieldnames)

    # write headers if needed
    if not file_exists:
        writer.writeheader()
    if not failed_exists:
        failed_writer.writeheader()

    count = 0
    for index, row in df.iterrows():
        if count == 3000:
            break
        count += 1
        url = row['Link']
        print(f"Processing ({index+1}/{len(df)}): {url}")
        try:
            a = newspaper.Article(url, language='en')
            a.download()
            a.parse()

            article_data = {
                "link":           url,
                "title":          a.title,
                "text":           a.text,
                "published_date": a.publish_date.isoformat() if a.publish_date else "",
                "keywords":       "; ".join(a.keywords),
            }

            # append this row to articles.csv
            writer.writerow(article_data)
            out_csv.flush()

        except Exception:
            print(f"  ✗ Failed: {url}")
            # log only the failed URL
            failed_writer.writerow({"link": url})
            fail_csv.flush()

        time.sleep(1)

print(f"Done. Articles appended to '{output_file}', failures logged to '{failed_file}'.")



