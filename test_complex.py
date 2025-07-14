#!/usr/bin/env python3
import asyncio
import sys
sys.path.append('src')
from research_mcp.content_analyzer.server import handle_call_tool

async def test_complex():
    # Biased F1 content
    biased_text = """
    Max Verstappen is absolutely the greatest driver of all time! Every single race proves he's completely superior to all other drivers. Lewis Hamilton is clearly past his prime and will never win another championship. The data obviously shows that Red Bull's dominance is totally unprecedented in F1 history. According to recent analysis, Verstappen's win rate is 85% this season.
    """
    
    result = await handle_call_tool("analyze_content", {
        "text": biased_text,
        "source_url": "https://f1blog.example.com/verstappen-goat"
    })
    
    print("🔍 COMPLEX CONTENT ANALYSIS:")
    print(result[0].text)

if __name__ == "__main__":
    asyncio.run(test_complex())
