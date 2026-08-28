import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai import BrowserConfig, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

async def main(url, question):
    cleaned_question = question.strip()
    print(f"DEBUG question repr: {repr(cleaned_question)}")
    
    bm25_filter = BM25ContentFilter(user_query=cleaned_question,bm25_threshold= 1)
    
    md_g = DefaultMarkdownGenerator(content_filter=bm25_filter)
    bc = BrowserConfig()
    rc = CrawlerRunConfig(markdown_generator=md_g)
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=rc)
        
    if result.success:
        return result.markdown.fit_markdown
    else:
        return "Error : Something gone wrong !"


def scrape_page(url,question):
    result = asyncio.run(main(url,question))
    return result 
