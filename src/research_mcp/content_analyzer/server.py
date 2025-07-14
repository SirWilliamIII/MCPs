#!/usr/bin/env python3
"""
Content Analyzer MCP Server
Analyzes text for sentiment, bias, credibility and key facts
"""
import asyncio
from typing import Any, Dict
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
import mcp.types as types

# NLP imports
import spacy
from textblob import TextBlob
import re
from collections import Counter

# Load spacy model (we'll handle errors gracefully)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Create the MCP server
server = Server("content-analyzer")

def analyze_text_advanced(text: str, source_url: str = "Unknown") -> Dict[str, Any]:
    """Advanced text analysis using multiple NLP techniques"""
    
    # Basic metrics
    words = text.split()
    sentences = text.split('.')
    
    # Sentiment analysis with TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    # Determine sentiment category
    if sentiment.polarity > 0.1:
        sentiment_label = "Positive"
    elif sentiment.polarity < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    # Entity extraction with spaCy (if available)
    entities = []
    if nlp:
        doc = nlp(text)
        entities = [{"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_)} 
                   for ent in doc.ents]
    
    # Bias detection patterns
    bias_indicators = detect_bias_patterns(text)
    
    # Readability assessment
    readability = assess_readability(text, words, sentences)
    
    # Key facts extraction
    key_facts = extract_key_facts(text)
    
    return {
        "basic_metrics": {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "character_count": len(text),
            "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 1)
        },
        "sentiment_analysis": {
            "label": sentiment_label,
            "polarity": round(sentiment.polarity, 3),
            "subjectivity": round(sentiment.subjectivity, 3),
            "confidence": "High" if abs(sentiment.polarity) > 0.3 else "Medium" if abs(sentiment.polarity) > 0.1 else "Low"
        },
        "entities": entities[:10],  # Limit to top 10 entities
        "bias_indicators": bias_indicators,
        "readability": readability,
        "key_facts": key_facts,
        "source": source_url,
        "analysis_version": "1.0"
    }

def detect_bias_patterns(text: str) -> Dict[str, Any]:
    """Detect potential bias indicators in text"""
    
    # Emotional language patterns
    emotional_words = [
        "amazing", "terrible", "incredible", "awful", "fantastic", "horrible",
        "brilliant", "stupid", "genius", "idiot", "perfect", "disaster"
    ]
    
    # Absolute language patterns  
    absolute_words = [
        "always", "never", "all", "none", "every", "completely", "totally",
        "absolutely", "definitely", "certainly", "obviously", "clearly"
    ]
    
    # Find matches
    text_lower = text.lower()
    emotional_matches = [word for word in emotional_words if word in text_lower]
    absolute_matches = [word for word in absolute_words if word in text_lower]
    
    # Calculate bias score
    total_words = len(text.split())
    bias_word_count = len(emotional_matches) + len(absolute_matches)
    bias_ratio = bias_word_count / max(total_words, 1)
    
    return {
        "emotional_language": emotional_matches,
        "absolute_language": absolute_matches,
        "bias_score": round(bias_ratio * 100, 1),
        "assessment": "High bias" if bias_ratio > 0.05 else "Medium bias" if bias_ratio > 0.02 else "Low bias"
    }

def assess_readability(text: str, words: list, sentences: list) -> Dict[str, Any]:
    """Simple readability assessment"""
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
    
    # Complexity indicators
    long_words = [word for word in words if len(word) > 6]
    complex_ratio = len(long_words) / max(len(words), 1)
    
    # Simple readability score (simplified Flesch-like)
    avg_sentence_length = len(words) / max(len(sentences), 1)
    readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * complex_ratio)
    
    # Readability level
    if readability_score >= 90:
        level = "Very Easy"
    elif readability_score >= 80:
        level = "Easy"
    elif readability_score >= 70:
        level = "Fairly Easy"
    elif readability_score >= 60:
        level = "Standard"
    elif readability_score >= 50:
        level = "Fairly Difficult"
    else:
        level = "Difficult"
    
    return {
        "score": round(readability_score, 1),
        "level": level,
        "avg_word_length": round(avg_word_length, 1),
        "complex_word_ratio": round(complex_ratio * 100, 1)
    }

def extract_key_facts(text: str) -> list:
    """Extract potential key facts from text"""
    
    # Look for number patterns (statistics, dates, percentages)
    number_patterns = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    
    # Look for quoted statements
    quotes = re.findall(r'"([^"]*)"', text)
    
    # Look for sentences with factual indicators
    factual_indicators = ['according to', 'study shows', 'research indicates', 'data reveals', 'statistics show']
    factual_sentences = []
    
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if any(indicator in sentence.lower() for indicator in factual_indicators):
            factual_sentences.append(sentence)
    
    return {
        "numbers_and_stats": number_patterns[:5],  # Top 5 numbers
        "quotes": quotes[:3],  # Top 3 quotes
        "factual_claims": factual_sentences[:3]  # Top 3 factual sentences
    }

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="analyze_content",
            description="Advanced text analysis: sentiment, bias, entities, readability, and key facts",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string", 
                        "description": "Text content to analyze"
                    },
                    "source_url": {
                        "type": "string", 
                        "description": "Source URL (optional)"
                    }
                },
                "required": ["text"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    """Handle tool calls"""
    if name == "analyze_content":
        text = arguments["text"]
        source_url = arguments.get("source_url", "Unknown")
        
        # Perform advanced analysis
        analysis = analyze_text_advanced(text, source_url)
        
        # Format results nicely
        result = f"""
🔍 CONTENT ANALYSIS REPORT
{'=' * 50}

📊 BASIC METRICS:
- Words: {analysis['basic_metrics']['word_count']}
- Sentences: {analysis['basic_metrics']['sentence_count']}
- Avg words/sentence: {analysis['basic_metrics']['avg_words_per_sentence']}

😊 SENTIMENT ANALYSIS:
- Overall sentiment: {analysis['sentiment_analysis']['label']}
- Polarity: {analysis['sentiment_analysis']['polarity']} (-1 to +1)
- Subjectivity: {analysis['sentiment_analysis']['subjectivity']} (0 to 1)
- Confidence: {analysis['sentiment_analysis']['confidence']}

⚖️ BIAS ASSESSMENT:
- Bias level: {analysis['bias_indicators']['assessment']}
- Bias score: {analysis['bias_indicators']['bias_score']}%
- Emotional words: {', '.join(analysis['bias_indicators']['emotional_language'][:5]) or 'None detected'}
- Absolute language: {', '.join(analysis['bias_indicators']['absolute_language'][:5]) or 'None detected'}

📚 READABILITY:
- Level: {analysis['readability']['level']}
- Score: {analysis['readability']['score']}/100
- Avg word length: {analysis['readability']['avg_word_length']} characters

🏷️ KEY ENTITIES:
{chr(10).join([f'• {ent["text"]} ({ent["label"]})' for ent in analysis['entities'][:5]]) or '• No entities detected'}

📝 KEY FACTS:
- Numbers/Stats: {', '.join(analysis['key_facts']['numbers_and_stats']) or 'None found'}
- Quotes: {len(analysis['key_facts']['quotes'])} found
- Factual claims: {len(analysis['key_facts']['factual_claims'])} found

🔗 Source: {analysis['source']}
"""
        
        return [types.TextContent(type="text", text=result)]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="content-analyzer",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
    
