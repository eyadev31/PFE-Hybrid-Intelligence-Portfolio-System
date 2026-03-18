from ml.news_collector import NewsCollector

collector = NewsCollector()

articles = collector.collect_news()

print("FINAL ARTICLES:", len(articles))

for a in articles[:5]:
    print(a["source"], "-", a["title"])