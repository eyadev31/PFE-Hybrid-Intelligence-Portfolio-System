import json

try:
    with open('agent5_live_with_key.txt', 'r', encoding='utf-16le') as f:
        text = f.read()
    json_str = text[text.find('{'):text.rfind('}')+1]
    
    # Save a UTF-8 copy of just the JSON for easy reading
    with open('agent5_live_with_key_json.txt', 'w', encoding='utf-8') as f:
        f.write(json_str)

    data = json.loads(json_str)

    meta = data.get('agent_metadata', {})
    print('\n--- LIVE METADATA WITH NEWSAPI ---')
    print('Data Quality:', meta.get('data_quality'))
    print('Articles Processed:', meta.get('articles_processed'))
    print('Sources Queried:', meta.get('sources_queried'))
    print('Models Used:', meta.get('models_used'))
    print('Execution Time (ms):', meta.get('execution_time_ms'))
except Exception as e:
    print('Error parsing output:', e)
