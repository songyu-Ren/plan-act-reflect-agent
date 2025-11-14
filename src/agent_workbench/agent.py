from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agent_workbench.llm.providers import LLMProvider, Message
from agent_workbench.memory.long_vector import VectorMemory
from agent_workbench.memory.short_sql import (
    MessageRecord,
    ReflectionRecord,
    ShortTermMemory,
    ToolEvent,
)
from agent_workbench.planner import Plan, PlanStep, Planner
from agent_workbench.reflection import ReflectionResult, Reflector
from agent_workbench.settings import Settings
from agent_workbench.telemetry import get_metrics
from agent_workbench.tools.fs import FilesystemTool
from agent_workbench.tools.python_runner import PythonRunner
from agent_workbench.tools.rag import RAGTool
from agent_workbench.tools.web import fetch_url
from agent_workbench.skills import SkillsRegistry, SkillContext
from agent_workbench.planner_hier import Manager
from agent_workbench.trace import TraceWriter
from agent_workbench.cost import CostTracker
from agent_workbench.hitl import GLOBAL_APPROVAL_STORE


class AgentStep(BaseModel):
    step_number: int
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Dict[str, Any]
    reflection: ReflectionResult
    timestamp: datetime


class AgentResult(BaseModel):
    status: str  # "success", "failure", "stopped"
    goal: str
    steps_taken: List[AgentStep]
    final_output: Optional[str] = None
    artifacts_paths: List[str] = []
    memory_updates: Dict[str, Any] = {}
    session_id: str
    run_id: Optional[str] = None


class Agent:
    def __init__(self, settings: Settings, llm_provider: LLMProvider):
        self.settings = settings
        self.llm_provider = llm_provider
        self.short_memory = ShortTermMemory(settings)
        self.vector_memory = VectorMemory(settings)
        self.metrics = get_metrics(settings)
        
        # Initialize tools
        self.tools = {
            "web": fetch_url,
            "fs": FilesystemTool(settings),
            "python": PythonRunner(settings),
            "rag": RAGTool(settings),
        }

        # Skills and planner
        self.skills = SkillsRegistry(settings)
        self.skills.load_builtins()
        self.manager = Manager(self.settings.skills.get("allowed", []), concurrency=self.settings.skills.get("concurrency", 1))

        # Trace and cost
        self.tracer = TraceWriter(self.settings.tracing.get("export_dir", "artifacts/traces"))
        self.costs = CostTracker()
        self.approvals = GLOBAL_APPROVAL_STORE
        
        self.planner = Planner(llm_provider, settings)
        self.reflector = Reflector(llm_provider, settings)
        
        # Track current session
        self.current_session_id: Optional[str] = None
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        await self.short_memory.initialize()
    
    async def run_task(
        self,
        goal: str,
        session_id: Optional[str] = None,
        max_steps: Optional[int] = None,
        constraints: Optional[List[str]] = None
    ) -> AgentResult:
        """Run a task with the agent loop"""
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        
        # Create session in memory
        await self.short_memory.create_session(session_id)
        
        # Record initial message
        await self.short_memory.add_message(MessageRecord(
            session_id=session_id,
            role="user",
            content=f"Goal: {goal}",
            timestamp=datetime.now()
        ))
        
        if max_steps is None:
            max_steps = self.settings.agent.max_steps
        
        steps_taken = []
        run_id = self.tracer.new_run() if self.settings.tracing.get("enabled", True) else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tracer.append(run_id, {"type": "plan_start", "goal": goal})
        current_state = f"Starting task with goal: {goal}"
        
        # Planning phase
        if self.settings.agent.planning_style == "plan_execute":
            plan = self.planner.plan(goal, current_state)
            await self._log_plan(plan, session_id)
        else:
            graph = self.manager.build_plan(goal)
            self.tracer.append(run_id, {"type": "plan_graph", "nodes": [graph.nodes[n]["data"].__dict__ for n in graph.nodes], "edges": list(graph.edges)})
        
        step_count = 0
        
        while step_count < max_steps:
            step_count += 1
            
            # Get next action
            if self.settings.agent.planning_style == "plan_execute" and step_count <= len(plan.steps):
                next_action = {
                    "tool_name": plan.steps[step_count - 1].tool_name,
                    "tool_input": plan.steps[step_count - 1].tool_input,
                    "rationale": plan.steps[step_count - 1].rationale
                }
            else:
                # Hierarchical plan execution: take next pending node
                pending_nodes = [n for n in graph.nodes if graph.nodes[n]["data"].status == "pending" and all(graph.nodes[p]["data"].status == "done" for p in graph.predecessors(n))]
                if pending_nodes:
                    node = graph.nodes[pending_nodes[0]]["data"]
                    next_action = {"tool_name": node.skill.split('.')[0], "skill": node.skill, "tool_input": node.args, "rationale": "hierarchical"}
                else:
                    next_action = None
                
                if next_action is None:
                    # Goal achieved
                    break
            
            # Execute tool or skill
            tool_name = next_action.get("tool_name")
            tool_input = next_action["tool_input"]
            skill_name = next_action.get("skill")
            
            # HITL approvals for risky actions
            risky_actions = {a.get("action"): a.get("reason") for a in self.settings.hitl.get("approvals", [])}
            if skill_name in risky_actions:
                item = self.approvals.create(skill_name, risky_actions[skill_name])
                self.tracer.append(run_id, {"type": "approval", "id": item.id, "action": item.action, "reason": item.reason})
                if self.llm_provider.__class__.__name__ == "NullProvider":
                    self.approvals.approve(item.id)
                else:
                    await asyncio.sleep(0.1)

            if skill_name:
                ctx = SkillContext(session_id=session_id, settings=self.settings)
                tool_result = self.skills.execute(skill_name, ctx, tool_input)
                self.metrics.record_skill_call(skill_name, "success" if tool_result.get("success") else "failure")
            else:
                tool_result = await self._execute_tool(tool_name, tool_input, session_id)

            self.tracer.append(run_id, {"type": "tool_call", "name": skill_name or tool_name, "args": tool_input, "result": tool_result})
            self.costs.add_steps(1)
            
            # Record tool event
            await self.short_memory.add_tool_event(ToolEvent(
                session_id=session_id,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_result if tool_result.get("success") else None,
                error=tool_result.get("error"),
                timestamp=datetime.now()
            ))
            
            # Reflect on result
            reflection = self.reflector.reflect(
                goal=goal,
                step_history=[{"tool": s.tool_name, "result": s.tool_result} for s in steps_taken],
                current_state=current_state,
                tool_result=tool_result
            )
            
            # Record reflection
            await self.short_memory.add_reflection(ReflectionRecord(
                session_id=session_id,
                step_number=step_count,
                reflection_text=reflection.reflection_text,
                usefulness_score=reflection.usefulness_score,
                memory_updates=reflection.memory_updates,
                timestamp=datetime.now()
            ))
            
            # Create step record
            step = AgentStep(
                step_number=step_count,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=tool_result,
                reflection=reflection,
                timestamp=datetime.now()
            )
            
            steps_taken.append(step)
            
            # Update metrics
            self.metrics.record_tool_call(tool_name)
            self.metrics.record_agent_step()
            
            # Update current state
            current_state = self._update_current_state(current_state, tool_result, reflection)
            
            # Check if we should continue
            if not reflection.should_continue:
                break
            
            # Small delay to prevent rapid execution
            await asyncio.sleep(0.1)
        
        # Determine final status
        if step_count >= max_steps:
            status = "stopped"
            final_output = f"Task stopped after {max_steps} steps"
        elif any(s.reflection.usefulness_score > 0.8 for s in steps_taken):
            status = "success"
            final_output = "Task completed successfully"
        else:
            status = "failure"
            final_output = "Task failed to achieve goal"
        
        # Collect artifacts
        artifacts_paths = self._collect_artifacts(session_id)
        
        # Collect memory updates
        memory_updates = {}
        for step in steps_taken:
            if step.reflection.memory_updates:
                memory_updates.update(step.reflection.memory_updates)
        
        result = AgentResult(
            status=status,
            goal=goal,
            steps_taken=steps_taken,
            final_output=final_output,
            artifacts_paths=artifacts_paths,
            memory_updates=memory_updates,
            session_id=session_id,
            run_id=run_id
        )
        
        # Record final message
        await self.short_memory.add_message(MessageRecord(
            session_id=session_id,
            role="assistant",
            content=f"Task completed with status: {status}. {final_output}",
            timestamp=datetime.now(),
            metadata={"status": status, "steps": step_count}
        ))
        
        self.tracer.append(run_id, {"type": "done", "status": status, "cost": self.costs.snapshot()})
        try:
            self.metrics.record_run(status)
            self.metrics.add_cost("steps", self.costs.snapshot().get("steps", 0))
        except Exception:
            pass
        return result
    
    async def chat(self, session_id: str, user_text: str) -> str:
        """Handle chat interaction"""
        
        # Ensure session exists
        await self.short_memory.create_session(session_id)
        
        # Record user message
        await self.short_memory.add_message(MessageRecord(
            session_id=session_id,
            role="user",
            content=user_text,
            timestamp=datetime.now()
        ))
        
        # Get conversation history
        history = await self.short_memory.get_session_history(session_id, limit=10)
        
        # Build context
        context = self._build_chat_context(history)
        
        # Generate response
        messages = [
            Message(role="system", content="You are a helpful AI assistant. Use available tools when needed to answer questions."),
            Message(role="user", content=f"Context: {context}\n\nUser: {user_text}")
        ]
        
        response = self.llm_provider.generate(messages)
        
        # Record assistant message
        await self.short_memory.add_message(MessageRecord(
            session_id=session_id,
            role="assistant",
            content=response.content,
            timestamp=datetime.now()
        ))
        
        return response.content
    
    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute a tool and return the result"""
        
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys())
            }
        
        tool = self.tools[tool_name]
        
        try:
            if tool_name == "web":
                url = tool_input.get("url")
                max_chars = tool_input.get("max_chars", 10000)
                if not url:
                    return {"success": False, "error": "URL required for web tool"}
                result = await tool(url, max_chars)
                
            elif tool_name == "fs":
                action = tool_input.get("action", "read")
                path = tool_input.get("path", "")
                
                if action == "read":
                    result = tool.read(path)
                elif action == "write":
                    content = tool_input.get("content", "")
                    encoding = tool_input.get("encoding", "utf-8")
                    result = tool.write(path, content, encoding)
                elif action == "list":
                    result = tool.list_dir(path)
                elif action == "delete":
                    result = tool.delete(path)
                elif action == "exists":
                    result = tool.exists(path)
                elif action == "mkdir":
                    result = tool.create_dir(path)
                else:
                    result = {"success": False, "error": f"Unknown fs action: {action}"}
                
            elif tool_name == "python":
                code = tool_input.get("code", "")
                if not code:
                    return {"success": False, "error": "Code required for python tool"}
                
                # Validate code first
                validation = tool.validate_code(code)
                if not validation["valid"]:
                    return {"success": False, "error": validation["reason"]}
                
                result = tool.run(code)
                
            elif tool_name == "rag":
                query = tool_input.get("query", "")
                k = tool_input.get("k")
                if not query:
                    return {"success": False, "error": "Query required for rag tool"}
                result = tool.search(query, k)
                
            else:
                result = {"success": False, "error": f"Tool {tool_name} not implemented"}
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Tool execution error: {str(e)}"
            }
    
    async def _log_plan(self, plan: Plan, session_id: str) -> None:
        """Log the plan to memory"""
        plan_text = f"Plan for goal '{plan.goal}':\n{plan.overall_rationale}\n\n"
        for step in plan.steps:
            plan_text += f"Step {step.step_number}: {step.tool_name} - {step.rationale}\n"
        
        await self.short_memory.add_message(MessageRecord(
            session_id=session_id,
            role="system",
            content=plan_text,
            timestamp=datetime.now(),
            metadata={"type": "plan", "steps": len(plan.steps)}
        ))
    
    def _format_step_history(self, steps: List[AgentStep]) -> str:
        """Format step history for context"""
        history = []
        for step in steps:
            success = "✓" if step.tool_result.get("success") else "✗"
            history.append(f"{success} {step.tool_name}: {step.reflection.reflection_text[:100]}...")
        return "\n".join(history)
    
    def _update_current_state(self, current_state: str, tool_result: Dict[str, Any], reflection: ReflectionResult) -> str:
        """Update the current state based on tool result and reflection"""
        status = "success" if tool_result.get("success") else "failed"
        tool_summary = f"Last action {status}: {reflection.reflection_text[:200]}..."
        return f"{current_state}\n{tool_summary}"
    
    def _build_chat_context(self, history: List[MessageRecord]) -> str:
        """Build context from chat history"""
        if not history:
            return "No previous conversation."
        
        context = "Recent conversation:\n"
        for msg in history[-5:]:  # Last 5 messages
            context += f"{msg.role}: {msg.content[:100]}...\n"
        
        return context
    
    def _collect_artifacts(self, session_id: str) -> List[str]:
        """Collect paths to artifacts created during the session"""
        workspace = Path(self.settings.paths.workspace_dir)
        artifacts = []
        
        # Find files created during this session (simplified)
        for file_path in workspace.rglob("*"):
            if file_path.is_file():
                artifacts.append(str(file_path.relative_to(workspace)))
        
        return artifacts
