from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agent_workbench.llm.providers import LLMProvider, Message
from agent_workbench.settings import Settings


class ReflectionResult(BaseModel):
    usefulness_score: float  # 0.0 to 1.0
    reflection_text: str
    memory_updates: Dict[str, Any]
    should_continue: bool
    next_action: Optional[str] = None


class Reflector:
    def __init__(self, llm_provider: LLMProvider, settings: Settings):
        self.llm_provider = llm_provider
        self.settings = settings
    
    def reflect(
        self,
        goal: str,
        step_history: List[Dict[str, Any]],
        current_state: str,
        tool_result: Dict[str, Any]
    ) -> ReflectionResult:
        """Reflect on the current state and decide next actions"""
        
        prompt = self._build_reflection_prompt(goal, step_history, current_state, tool_result)
        
        messages = [
            Message(role="system", content="You are an AI reflection agent that assesses progress toward goals and decides next actions."),
            Message(role="user", content=prompt)
        ]
        
        response = self.llm_provider.generate(messages)
        
        return self._parse_reflection_response(response.content)
    
    def _build_reflection_prompt(
        self,
        goal: str,
        step_history: List[Dict[str, Any]],
        current_state: str,
        tool_result: Dict[str, Any]
    ) -> str:
        
        history_text = ""
        for i, step in enumerate(step_history, 1):
            history_text += f"Step {i}: {step.get('tool', 'unknown')} - {step.get('result', 'no result')}\n"
        
        tool_success = tool_result.get("success", False)
        tool_error = tool_result.get("error", "")
        
        prompt = f"""Reflect on the current state of this agent task and decide the next action.

GOAL: {goal}

STEP HISTORY:
{history_text}

CURRENT STATE:
{current_state}

LAST TOOL RESULT:
Success: {tool_success}
Error: {tool_error}
Result: {tool_result}

REFLECTION TASKS:
1. Assess how useful the last tool result was for achieving the goal (0.0-1.0)
2. Determine if the goal has been achieved
3. Decide whether to continue or stop
4. If continuing, suggest the next action
5. Identify any key insights or learnings
6. Suggest memory updates if relevant

Provide your reflection in this format:

USEFULNESS: [0.0-1.0 score]
GOAL_ACHIEVED: [yes/no]
SHOULD_CONTINUE: [yes/no]
NEXT_ACTION: [suggested next action or "none"]

REFLECTION:
[Your detailed analysis and reasoning]

MEMORY_UPDATES:
[key1: value1, key2: value2, ... or "none"]"""
        
        return prompt
    
    def _parse_reflection_response(self, response: str) -> ReflectionResult:
        """Parse the reflection response"""
        
        lines = response.split('\n')
        
        usefulness_score = 0.5
        goal_achieved = False
        should_continue = True
        next_action = None
        reflection_text = ""
        memory_updates = {}
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("USEFULNESS:"):
                try:
                    usefulness_score = float(line.replace("USEFULNESS:", "").strip())
                    usefulness_score = max(0.0, min(1.0, usefulness_score))
                except:
                    pass
            
            elif line.startswith("GOAL_ACHIEVED:"):
                goal_achieved = "yes" in line.lower()
            
            elif line.startswith("SHOULD_CONTINUE:"):
                should_continue = "yes" in line.lower()
            
            elif line.startswith("NEXT_ACTION:"):
                next_action_text = line.replace("NEXT_ACTION:", "").strip()
                if next_action_text.lower() not in ["none", "", "stop"]:
                    next_action = next_action_text
            
            elif line.startswith("REFLECTION:"):
                current_section = "reflection"
            
            elif line.startswith("MEMORY_UPDATES:"):
                current_section = "memory"
                memory_text = line.replace("MEMORY_UPDATES:", "").strip()
                if memory_text.lower() not in ["none", "", "no updates"]:
                    try:
                        # Try to parse as simple key:value pairs
                        pairs = memory_text.split(',')
                        for pair in pairs:
                            if ':' in pair:
                                key, value = pair.split(':', 1)
                                memory_updates[key.strip()] = value.strip()
                    except:
                        pass
            
            elif current_section == "reflection" and line:
                if reflection_text:
                    reflection_text += "\n" + line
                else:
                    reflection_text = line
        
        # Adjust should_continue based on goal achievement
        if goal_achieved:
            should_continue = False
        
        return ReflectionResult(
            usefulness_score=usefulness_score,
            reflection_text=reflection_text or response,
            memory_updates=memory_updates,
            should_continue=should_continue,
            next_action=next_action
        )
    
    def summarize_session(self, goal: str, steps: List[Dict[str, Any]]) -> str:
        """Generate a summary of the session"""
        
        step_count = len(steps)
        successful_steps = sum(1 for step in steps if step.get("success", False))
        
        summary = f"Session Summary:\n"
        summary += f"Goal: {goal}\n"
        summary += f"Total steps: {step_count}\n"
        summary += f"Successful steps: {successful_steps}\n"
        summary += f"Success rate: {successful_steps/step_count*100:.1f}%\n"
        
        if steps:
            summary += "\nKey actions taken:\n"
            for i, step in enumerate(steps, 1):
                tool = step.get("tool", "unknown")
                success = "✓" if step.get("success", False) else "✗"
                summary += f"  {i}. {success} {tool}\n"
        
        return summary