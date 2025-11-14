from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agent_workbench.llm.providers import LLMProvider, Message
from agent_workbench.settings import Settings


class PlanStep(BaseModel):
    step_number: int
    tool_name: str
    tool_input: Dict[str, Any]
    rationale: str
    expected_outcome: str


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]
    overall_rationale: str


class Planner:
    def __init__(self, llm_provider: LLMProvider, settings: Settings):
        self.llm_provider = llm_provider
        self.settings = settings
    
    def plan(self, goal: str, context: str = "", available_tools: Optional[List[str]] = None) -> Plan:
        """Generate a plan for achieving the goal"""
        if available_tools is None:
            available_tools = self.settings.agent.allow_tools
        
        prompt = self._build_planning_prompt(goal, context, available_tools)
        
        messages = [
            Message(role="system", content="You are an AI planning agent that creates step-by-step plans to achieve goals using available tools."),
            Message(role="user", content=prompt)
        ]
        
        response = self.llm_provider.generate(messages)
        
        # Parse the response to extract plan
        return self._parse_plan_response(response.content, goal)
    
    def _build_planning_prompt(self, goal: str, context: str, available_tools: List[str]) -> str:
        tool_descriptions = {
            "web": "Fetch and clean content from web URLs",
            "fs": "Read/write files in the workspace directory",
            "python": "Execute Python code in a sandboxed environment",
            "rag": "Search the knowledge corpus for relevant information"
        }
        
        prompt = f"""Create a step-by-step plan to achieve this goal: {goal}

Context: {context}

Available tools:
{chr(10).join([f"- {tool}: {tool_descriptions.get(tool, 'No description available')}" for tool in available_tools])}

Planning requirements:
1. Break down the goal into specific, actionable steps
2. Each step should use exactly one tool
3. Provide clear tool inputs for each step
4. Explain the rationale for each step
5. Describe the expected outcome
6. Consider dependencies between steps
7. Include verification steps where appropriate

Return your plan in this format:

OVERALL RATIONALE:
[Explain your overall approach and strategy]

STEPS:
Step 1: [Tool Name]
Input: [Tool input as JSON]
Rationale: [Why this step is needed]
Expected: [What you expect to achieve]

Step 2: [Tool Name]
Input: [Tool input as JSON]
Rationale: [Why this step is needed]
Expected: [What you expect to achieve]

[Continue for all needed steps...]
"""
        return prompt
    
    def _parse_plan_response(self, response: str, goal: str) -> Plan:
        """Parse the LLM response into a structured plan"""
        steps = []
        overall_rationale = ""
        
        lines = response.split('\n')
        current_step = None
        step_data = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("OVERALL RATIONALE:"):
                overall_rationale = line.replace("OVERALL RATIONALE:", "").strip()
            elif line.startswith("Step "):
                # Save previous step if exists
                if current_step and step_data:
                    try:
                        step = PlanStep(
                            step_number=current_step,
                            tool_name=step_data["tool"],
                            tool_input=step_data["input"],
                            rationale=step_data["rationale"],
                            expected_outcome=step_data["expected"]
                        )
                        steps.append(step)
                    except:
                        pass  # Skip malformed steps
                
                # Start new step
                step_text = line.split(":", 1)[1].strip() if ":" in line else ""
                current_step = len(steps) + 1
                step_data = {"tool": step_text, "input": {}, "rationale": "", "expected": ""}
            
            elif line.startswith("Input:") and current_step:
                input_text = line.replace("Input:", "").strip()
                try:
                    # Try to parse as JSON
                    import json
                    step_data["input"] = json.loads(input_text)
                except:
                    step_data["input"] = {"query": input_text}
            
            elif line.startswith("Rationale:") and current_step:
                step_data["rationale"] = line.replace("Rationale:", "").strip()
            
            elif line.startswith("Expected:") and current_step:
                step_data["expected"] = line.replace("Expected:", "").strip()
        
        # Add final step
        if current_step and step_data:
            try:
                step = PlanStep(
                    step_number=current_step,
                    tool_name=step_data["tool"],
                    tool_input=step_data["input"],
                    rationale=step_data["rationale"],
                    expected_outcome=step_data["expected"]
                )
                steps.append(step)
            except:
                pass
        
        return Plan(
            goal=goal,
            steps=steps,
            overall_rationale=overall_rationale
        )
    
    def suggest_next_step(self, goal: str, history: str, last_result: str) -> Optional[Dict[str, Any]]:
        """Suggest the next step based on current state"""
        prompt = f"""Based on the current state, suggest the next action to progress toward the goal.

Goal: {goal}

History:
{history}

Last result: {last_result}

Available tools: {', '.join(self.settings.agent.allow_tools)}

Suggest the next tool to use and its input. If the goal appears to be achieved, respond with "GOAL_ACHIEVED".

Format your response as:
TOOL: [tool name]
INPUT: [tool input as JSON]
RATIONALE: [brief explanation]"""
        
        messages = [
            Message(role="system", content="You are an AI that suggests the next action to achieve a goal."),
            Message(role="user", content=prompt)
        ]
        
        response = self.llm_provider.generate(messages)
        
        if "GOAL_ACHIEVED" in response.content:
            return None
        
        # Parse tool suggestion
        lines = response.content.split('\n')
        tool_name = ""
        tool_input = {}
        rationale = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("INPUT:"):
                input_text = line.replace("INPUT:", "").strip()
                try:
                    import json
                    tool_input = json.loads(input_text)
                except:
                    tool_input = {"query": input_text}
            elif line.startswith("RATIONALE:"):
                rationale = line.replace("RATIONALE:", "").strip()
        
        if tool_name:
            return {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "rationale": rationale
            }
        
        return None