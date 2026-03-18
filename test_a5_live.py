import json
from agents.agent5_news import Agent5NewsIntelligence

print("Starting Agent 5 (Mock=False)...")
a = Agent5NewsIntelligence()
o = a.run(mock=False)

print('\n\n--- RESULTS ---')
print(f"Data Quality: {o['agent_metadata']['data_quality']}")
print(f"Articles Processed: {o['article_count']}")
print(f"Sources Queried: {o['agent_metadata']['sources_queried']}")
print(f"Market Signal: {o['market_signal']['signal_type']}")
print(f"Narrative Preview: {o['market_signal'].get('narrative', '')[:150]}...")
print('\nTop 3 Headlines:')
for art in o['articles'][:3]:
    print(f" - {art['title']}")
