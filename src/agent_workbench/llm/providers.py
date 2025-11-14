from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import httpx
from pydantic import BaseModel

from agent_workbench.settings import LLMConfig


class Message(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None


class LLMProvider(ABC):
    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        pass

    @abstractmethod
    async def agenerate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        pass

    @abstractmethod
    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        pass

    @abstractmethod
    async def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[str]:
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )

    def generate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[m.dict() for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=response.usage.dict() if response.usage else None,
            model=response.model
        )

    async def agenerate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": [m.dict() for m in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
            model=data.get("model")
        )

    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        stream = client.chat.completions.create(
            model=self.config.model,
            messages=[m.dict() for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": self.config.model,
                "messages": [m.dict() for m in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = eval(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except:
                        continue


class AzureProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.endpoint = config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = config.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.endpoint or not self.api_key:
            raise ValueError("Azure OpenAI endpoint and API key required")
        
        self.client = httpx.AsyncClient(
            base_url=f"{self.endpoint}/openai/deployments/{config.model}",
            headers={"api-key": self.api_key},
            timeout=30.0
        )

    def generate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        import openai
        client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-15-preview"
        )
        
        response = client.chat.completions.create(
            model=self.config.model,
            messages=[m.dict() for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=response.usage.dict() if response.usage else None,
            model=response.model
        )

    async def agenerate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        response = await self.client.post(
            "/chat/completions",
            json={
                "messages": [m.dict() for m in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                **kwargs
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            usage=data.get("usage"),
            model=data.get("model")
        )

    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        import openai
        client = openai.AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version="2024-02-15-preview"
        )
        
        stream = client.chat.completions.create(
            model=self.config.model,
            messages=[m.dict() for m in messages],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json={
                "messages": [m.dict() for m in messages],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": True,
                **kwargs
            }
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = eval(data)
                        if chunk["choices"][0]["delta"].get("content"):
                            yield chunk["choices"][0]["delta"]["content"]
                    except:
                        continue


class OllamaProvider(LLMProvider):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    def generate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model")
        )

    async def agenerate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.config.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model")
        )

    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self.config.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            },
            timeout=30.0
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        data = eval(line)
                        if "message" in data and data["message"].get("content"):
                            yield data["message"]["content"]
                    except:
                        continue

    async def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[str]:
        async with self.client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.config.model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = eval(line)
                        if "message" in data and data["message"].get("content"):
                            yield data["message"]["content"]
                    except:
                        continue


class NullProvider(LLMProvider):
    """Null provider for testing and CI environments"""
    
    def generate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        # Get the last message to understand the context
        last = messages[-1] if messages else None
        last_message = last.content if hasattr(last, 'content') else (last.get('content') if isinstance(last, dict) else "")
        
        # Return contextually appropriate responses for different scenarios
        # Check for planning first (more specific patterns)
        if "create a step-by-step plan" in last_message.lower() or ("plan" in last_message.lower() and "step" in last_message.lower()):
            return self._generate_plan_response(last_message)
        elif "suggest the next action" in last_message.lower() or "suggest the next tool" in last_message.lower():
            return self._generate_next_step_response(last_message)
        elif "reflect" in last_message.lower() or "assess progress" in last_message.lower():
            return self._generate_reflection_response(last_message)
        else:
            return LLMResponse(
                content="[NULL PROVIDER] This is a simulated response for testing purposes.",
                usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
                model="null-model"
            )

    def _generate_plan_response(self, prompt: str) -> LLMResponse:
        """Generate a realistic plan response for testing"""
        plan_content = """OVERALL RATIONALE:
I will create a simple plan to achieve the goal using available tools. This is a simulated response for testing purposes.

STEPS:
Step 1: web
Input: {"url": "https://example.com", "max_chars": 5000}
Rationale: Start by gathering information from a reliable source
Expected: Get relevant content to work with

Step 2: python  
Input: {"code": "print('Hello from null provider test')"}
Rationale: Process the gathered information with Python
Expected: Demonstrate code execution capability

Step 3: fs
Input: {"action": "write", "path": "test_output.txt", "content": "Test completed successfully"}
Rationale: Save the results to a file for persistence
Expected: Create a file with the test results"""
        
        return LLMResponse(
            content=plan_content,
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            model="null-model"
        )

    def _generate_next_step_response(self, prompt: str) -> LLMResponse:
        """Generate a realistic next step response for testing"""
        # For testing, always return a tool suggestion
        # In a real scenario, we would check if the goal is achieved and return GOAL_ACHIEVED
        next_step_content = """TOOL: web
INPUT: {"url": "https://example.com", "max_chars": 5000}
RATIONALE: Gather initial information to start working toward the goal"""
        
        return LLMResponse(
            content=next_step_content,
            usage={"prompt_tokens": 30, "completion_tokens": 25, "total_tokens": 55},
            model="null-model"
        )

    def _generate_reflection_response(self, prompt: str) -> LLMResponse:
        """Generate a realistic reflection response for testing"""
        reflection_content = """USEFULNESS: 0.7
GOAL_ACHIEVED: no
SHOULD_CONTINUE: yes
NEXT_ACTION: Continue with next tool execution

REFLECTION:
The previous step was moderately useful. I can see progress toward the goal but more work is needed. This is a simulated reflection for testing purposes.

MEMORY_UPDATES:
progress: ongoing, confidence: moderate"""
        
        return LLMResponse(
            content=reflection_content,
            usage={"prompt_tokens": 40, "completion_tokens": 60, "total_tokens": 100},
            model="null-model"
        )

    async def agenerate(self, messages: List[Message], **kwargs: Any) -> LLMResponse:
        return self.generate(messages, **kwargs)

    def stream(self, messages: List[Message], **kwargs: Any) -> Iterator[str]:
        response = self.generate(messages, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "

    async def astream(self, messages: List[Message], **kwargs: Any) -> AsyncIterator[str]:
        response = self.generate(messages, **kwargs)
        words = response.content.split()
        for word in words:
            yield word + " "


def get_provider(config: LLMConfig) -> LLMProvider:
    provider_map = {
        "openai": OpenAIProvider,
        "azure": AzureProvider,
        "ollama": OllamaProvider,
        "null": NullProvider,
    }
    
    provider_class = provider_map.get(config.provider.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    return provider_class(config)
