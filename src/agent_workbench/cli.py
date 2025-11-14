from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from agent_workbench.agent import Agent
from agent_workbench.llm.providers import get_provider
from agent_workbench.logging import setup_logging
from agent_workbench.settings import Settings
from agent_workbench.tools.rag import RAGTool


@click.group()
@click.option('--config', '-c', help='Path to config file')
@click.pass_context
def main(ctx, config):
    """Agent Workbench CLI"""
    settings = Settings.load(config)
    settings.ensure_directories()
    
    logger = setup_logging(settings)
    llm_provider = get_provider(settings.llm)
    agent = Agent(settings, llm_provider)
    
    ctx.ensure_object(dict)
    ctx.obj['settings'] = settings
    ctx.obj['logger'] = logger
    ctx.obj['agent'] = agent
    ctx.obj['llm_provider'] = llm_provider


@main.command()
@click.argument('corpus_path', type=click.Path(exists=True))
@click.pass_context
def ingest(ctx, corpus_path):
    """Ingest documents from corpus directory"""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']
    
    logger.info(f"Ingesting documents from {corpus_path}")
    
    try:
        # Initialize agent
        agent = ctx.obj['agent']
        asyncio.run(agent.initialize())
        
        # Use RAG tool for ingestion
        rag_tool = RAGTool(settings)
        result = rag_tool.ingest_corpus()
        
        if result['success']:
            logger.info(f"Successfully ingested {result['ingested_count']} documents")
            click.echo(f"✓ Ingested {result['ingested_count']} documents")
        else:
            logger.error(f"Ingestion failed: {result['error']}")
            click.echo(f"✗ Ingestion failed: {result['error']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@main.command()
@click.option('--session', '-s', required=True, help='Session ID')
@click.pass_context
def chat(ctx, session):
    """Interactive chat with the agent"""
    logger = ctx.obj['logger']
    agent = ctx.obj['agent']
    
    logger.info(f"Starting chat session: {session}")
    click.echo(f"Chat session: {session}")
    click.echo("Type 'quit' or 'exit' to end the session")
    click.echo("-" * 40)
    
    try:
        asyncio.run(agent.initialize())
        
        while True:
            user_input = click.prompt("You", type=str)
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                response = asyncio.run(agent.chat(session, user_input))
                click.echo(f"Agent: {response}")
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                click.echo(f"Error: {e}")
                
    except KeyboardInterrupt:
        click.echo("\nChat session ended by user")
    except Exception as e:
        logger.error(f"Chat session error: {e}")
        click.echo(f"Error: {e}")


@main.command()
@click.option('--goal', '-g', required=True, help='Task goal')
@click.option('--session', '-s', help='Session ID (auto-generated if not provided)')
@click.option('--max-steps', '-n', default=10, help='Maximum number of steps')
@click.option('--output', '-o', help='Output file for results')
@click.pass_context
def run(ctx, goal, session, max_steps, output):
    """Run a task with the agent"""
    logger = ctx.obj['logger']
    agent = ctx.obj['agent']
    
    logger.info(f"Running task: {goal}")
    click.echo(f"Running task: {goal}")
    click.echo(f"Max steps: {max_steps}")
    if session:
        click.echo(f"Session: {session}")
    
    try:
        asyncio.run(agent.initialize())
        
        # Run the task
        result = asyncio.run(agent.run_task(
            goal=goal,
            session_id=session,
            max_steps=max_steps
        ))
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo(f"Task Status: {result.status.upper()}")
        click.echo(f"Steps Taken: {len(result.steps_taken)}")
        click.echo(f"Session ID: {result.session_id}")
        
        if result.final_output:
            click.echo(f"Final Output: {result.final_output}")
        
        if result.artifacts_paths:
            click.echo(f"Artifacts: {len(result.artifacts_paths)} files created")
            for path in result.artifacts_paths:
                click.echo(f"  - {path}")
        
        if result.memory_updates:
            click.echo(f"Memory Updates: {len(result.memory_updates)} entries")
        
        # Save results if output file specified
        if output:
            output_data = {
                "goal": result.goal,
                "status": result.status,
                "steps_taken": len(result.steps_taken),
                "final_output": result.final_output,
                "artifacts_paths": result.artifacts_paths,
                "memory_updates": result.memory_updates,
                "session_id": result.session_id
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            click.echo(f"\nResults saved to: {output}")
        
        # Show step summary
        if result.steps_taken:
            click.echo("\nStep Summary:")
            for step in result.steps_taken:
                success = "✓" if step.tool_result.get("success") else "✗"
                click.echo(f"  {success} Step {step.step_number}: {step.tool_name} (usefulness: {step.reflection.usefulness_score:.2f})")
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@main.command()
@click.pass_context
def eval(ctx):
    """Run evaluation tests"""
    logger = ctx.obj['logger']
    settings = ctx.obj['settings']
    
    logger.info("Running evaluation tests")
    click.echo("Running evaluation tests...")
    
    async def run_tests():
        try:
            agent = ctx.obj['agent']
            await agent.initialize()
            
            # Test 1: Basic tool functionality
            click.echo("Test 1: Tool functionality...")
            
            # Test filesystem tool
            fs_tool = agent.tools['fs']
            result = fs_tool.write("test.txt", "Hello, World!")
            assert result['success'], "File write failed"
            
            result = fs_tool.read("test.txt")
            assert result['content'] == "Hello, World!", "File read failed"
            
            # Test Python runner
            python_tool = agent.tools['python']
            result = python_tool.run("print('Hello from Python')")
            assert result['success'], "Python execution failed"
            assert "Hello from Python" in result['stdout'], "Python output incorrect"
            
            # Test RAG search (if corpus exists)
            rag_tool = agent.tools['rag']
            if settings.llm.provider != "null":
                result = rag_tool.search("test query")
                assert 'results' in result, "RAG search failed"
            
            click.echo("✓ Tool tests passed")
            
            # Test 2: Memory functionality
            click.echo("Test 2: Memory functionality...")
            
            from agent_workbench.memory.short_sql import MessageRecord
            from datetime import datetime
            
            test_session = "eval_test"
            await agent.short_memory.create_session(test_session)
            
            message = MessageRecord(
                session_id=test_session,
                role="user",
                content="Test message",
                timestamp=datetime.now()
            )
            
            await agent.short_memory.add_message(message)
            history = await agent.short_memory.get_session_history(test_session)
            assert len(history) > 0, "Memory storage failed"
            
            click.echo("✓ Memory tests passed")
            
            # Test 3: Agent task execution
            click.echo("Test 3: Agent task execution...")
            
            result = await agent.run_task(
                goal="Create a simple text file with greeting",
                max_steps=3
            )
            
            assert result.status in ["success", "failure", "stopped"], f"Invalid task status: {result.status}"
            assert len(result.steps_taken) > 0, "No steps taken"
            
            click.echo("✓ Agent tests passed")
            
            click.echo("\n✓ All evaluation tests passed!")
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            click.echo(f"✗ Evaluation failed: {e}")
            import sys
            sys.exit(1)
    
    asyncio.run(run_tests())


@main.command()
@click.pass_context
def status(ctx):
    """Show system status"""
    settings = ctx.obj['settings']
    logger = ctx.obj['logger']
    
    logger.info("Checking system status")
    click.echo("Agent Workbench Status")
    click.echo("=" * 30)
    
    click.echo(f"Config: {settings.llm.provider} provider, {settings.agent.max_steps} max steps")
    click.echo(f"Workspace: {settings.paths.workspace_dir}")
    click.echo(f"Database: {settings.paths.sqlite_db}")
    click.echo(f"Vector Index: {settings.paths.vector_index_dir}")
    
    # Check directories
    paths_to_check = [
        settings.paths.workspace_dir,
        settings.paths.sqlite_db,
        settings.paths.vector_index_dir,
        "data/corpus"
    ]
    
    click.echo("\nDirectory Status:")
    for path in paths_to_check:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                size = sum(f.stat().st_size for f in path_obj.rglob('*') if f.is_file())
                click.echo(f"  ✓ {path} ({size} bytes)")
            else:
                click.echo(f"  ✓ {path} ({path_obj.stat().st_size} bytes)")
        else:
            click.echo(f"  ✗ {path} (not found)")
    
    click.echo("\nTools Available:")
    agent = ctx.obj['agent']
    for tool_name in agent.tools:
        click.echo(f"  - {tool_name}")


if __name__ == "__main__":
    main()