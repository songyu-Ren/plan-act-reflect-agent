from __future__ import annotations

import resource
import subprocess
import tempfile
from typing import Any, Dict

from agent_workbench.settings import Settings


class PythonRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.timeout = 30  # seconds
        self.max_memory_mb = 512  # MB
        self.max_output_bytes = 100000  # ~100KB
    
    def run(self, code: str) -> Dict[str, Any]:
        """Run Python code in a sandboxed subprocess"""
        try:
            # Create temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Prepare subprocess
            cmd = [
                'python3', '-u', temp_file
            ]
            
            # Set resource limits (platform-specific)
            def set_limits():
                try:
                    # Memory limit (may not work on all systems)
                    resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_mb * 1024 * 1024, -1))
                    # CPU time limit
                    resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, -1))
                except:
                    pass  # Resource limits may not be available on all systems
            
            # Run subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=set_limits if hasattr(resource, 'setrlimit') else None,
                env={
                    'PYTHONPATH': '',
                    'PATH': '/usr/bin:/bin',
                    'HOME': '/tmp',
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONUNBUFFERED': '1'
                }
            )
            
            # Process output
            stdout = result.stdout
            stderr = result.stderr
            
            # Truncate output if too large
            if len(stdout) > self.max_output_bytes:
                stdout = stdout[:self.max_output_bytes] + "\n[OUTPUT TRUNCATED]"
            
            if len(stderr) > self.max_output_bytes:
                stderr = stderr[:self.max_output_bytes] + "\n[ERROR OUTPUT TRUNCATED]"
            
            return {
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timeout": False
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": "Code execution timed out",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "return_code": -1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "timeout": False
            }
        finally:
            # Cleanup temp file
            try:
                import os
                os.unlink(temp_file)
            except:
                pass
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """Basic code validation"""
        # Check for potentially dangerous imports
        dangerous_imports = [
            'os', 'sys', 'subprocess', 'socket', 'requests', 'urllib',
            'pickle', 'eval', 'exec', '__import__', 'importlib'
        ]
        
        for dangerous in dangerous_imports:
            if f"import {dangerous}" in code or f"from {dangerous}" in code:
                return {
                    "valid": False,
                    "reason": f"Import of '{dangerous}' is not allowed for security reasons"
                }
        
        # Check for file system operations
        file_operations = ['open(', 'file(', 'os.remove', 'os.unlink', 'os.rmdir']
        for op in file_operations:
            if op in code:
                return {
                    "valid": False,
                    "reason": f"File operation '{op}' is not allowed for security reasons"
                }
        
        # Check for network operations
        network_operations = ['socket', 'http', 'ftp', 'smtp']
        for op in network_operations:
            if op in code:
                return {
                    "valid": False,
                    "reason": f"Network operation '{op}' is not allowed for security reasons"
                }
        
        return {"valid": True, "reason": None}