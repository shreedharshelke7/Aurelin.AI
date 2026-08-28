from tavily import TavilyClient
import os
tavily_client = TavilyClient(api_key=os.getenv("tavily_api_key"))

def url_fetch(question):
    response = tavily_client.search(query=question, max_results=3)
    if not response["results"]:
        return []
    url = [results['url'] for results in response['results']]
    return url
