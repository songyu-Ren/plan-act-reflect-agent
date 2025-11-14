from __future__ import annotations

import re
from typing import Any, Dict, Optional

import httpx
import trafilatura
from readability import Document


async def fetch_url(url: str, max_chars: int = 10000) -> Dict[str, Any]:
    """Fetch and clean web content"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Try trafilatura first (better extraction)
            try:
                content = trafilatura.extract(response.text, include_comments=False, include_tables=False)
                if content:
                    title = trafilatura.extract_metadata(response.text).get("title", "")
                    return {
                        "content": content[:max_chars],
                        "title": title or url,
                        "source": url,
                        "method": "trafilatura"
                    }
            except:
                pass
            
            # Fallback to readability
            try:
                doc = Document(response.text)
                content = doc.summary()
                title = doc.title() or url
                
                # Clean HTML tags
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                return {
                    "content": content[:max_chars],
                    "title": title,
                    "source": url,
                    "method": "readability"
                }
            except:
                pass
            
            # Final fallback - just text content
            content = re.sub(r'<[^>]+>', '', response.text)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                "content": content[:max_chars],
                "title": url,
                "source": url,
                "method": "text_only"
            }
            
    except Exception as e:
        return {
            "error": f"Failed to fetch {url}: {str(e)}",
            "content": "",
            "title": "",
            "source": url,
            "method": "error"
        }


def clean_text(text: str, max_chars: int = 10000) -> str:
    """Clean and truncate text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text