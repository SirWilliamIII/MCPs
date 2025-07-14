#!/usr/bin/env python3
import asyncio
import json
import sys
sys.path.append('src')
from research_mcp.content_analyzer.server import handle_list_tools, handle_call_tool

async def test_tools():
    """Test our MCP server tools"""
    print("🧪 Testing MCP Content Analyzer...")
    
    # Test list_tools
    tools = await handle_list_tools()
    print(f"✅ Tools available: {len(tools)}")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description}")
    
    # Test analyze_content tool
    test_text = "This is a great article about F1 racing. Max Verstappen is the best driver!"
    result = await handle_call_tool("analyze_content", {"text": test_text})
    print(f"\n📊 Analysis Result:")
    print(result[0].text)

if __name__ == "__main__":
    asyncio.run(test_tools())
