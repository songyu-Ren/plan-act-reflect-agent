from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_workbench.settings import Settings


class FilesystemTool:
    def __init__(self, settings: Settings):
        self.workspace_dir = Path(settings.paths.workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_path(self, path: str) -> Path:
        """Ensure path is within workspace directory"""
        full_path = (self.workspace_dir / path).resolve()
        
        # Security check: ensure path is within workspace
        try:
            full_path.relative_to(self.workspace_dir)
        except ValueError:
            raise ValueError(f"Path {path} is outside workspace directory")
        
        return full_path
    
    def read(self, path: str) -> Dict[str, Any]:
        """Read file content"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.exists():
                return {"error": f"File not found: {path}", "content": None}
            
            if not full_path.is_file():
                return {"error": f"Path is not a file: {path}", "content": None}
            
            try:
                content = full_path.read_text(encoding='utf-8')
                return {
                    "content": content,
                    "path": path,
                    "size": len(content),
                    "encoding": "utf-8"
                }
            except UnicodeDecodeError:
                # Try with different encoding
                content = full_path.read_text(encoding='latin-1')
                return {
                    "content": content,
                    "path": path,
                    "size": len(content),
                    "encoding": "latin-1"
                }
                
        except Exception as e:
            return {"error": f"Failed to read {path}: {str(e)}", "content": None}
    
    def write(self, path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Write content to file"""
        try:
            full_path = self._validate_path(path)
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            full_path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "path": path,
                "size": len(content),
                "encoding": encoding
            }
            
        except Exception as e:
            return {"error": f"Failed to write {path}: {str(e)}", "success": False}
    
    def list_dir(self, path: str = "") -> Dict[str, Any]:
        """List directory contents"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.exists():
                return {"error": f"Directory not found: {path}", "items": []}
            
            if not full_path.is_dir():
                return {"error": f"Path is not a directory: {path}", "items": []}
            
            items = []
            for item in full_path.iterdir():
                item_info = {
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                }
                items.append(item_info)
            
            return {
                "path": path,
                "items": items,
                "total": len(items)
            }
            
        except Exception as e:
            return {"error": f"Failed to list {path}: {str(e)}", "items": []}
    
    def delete(self, path: str) -> Dict[str, Any]:
        """Delete file or directory"""
        try:
            full_path = self._validate_path(path)
            
            if not full_path.exists():
                return {"error": f"Path not found: {path}", "success": False}
            
            if full_path.is_file():
                full_path.unlink()
            elif full_path.is_dir():
                shutil.rmtree(full_path)
            
            return {
                "success": True,
                "path": path,
                "type": "file" if full_path.is_file() else "directory"
            }
            
        except Exception as e:
            return {"error": f"Failed to delete {path}: {str(e)}", "success": False}
    
    def exists(self, path: str) -> Dict[str, Any]:
        """Check if path exists"""
        try:
            full_path = self._validate_path(path)
            return {
                "exists": full_path.exists(),
                "path": path,
                "type": "file" if full_path.is_file() else "directory" if full_path.is_dir() else None
            }
        except ValueError:
            return {"exists": False, "path": path, "type": None}
    
    def create_dir(self, path: str) -> Dict[str, Any]:
        """Create directory"""
        try:
            full_path = self._validate_path(path)
            full_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "success": True,
                "path": path,
                "created": True
            }
            
        except Exception as e:
            return {"error": f"Failed to create directory {path}: {str(e)}", "success": False}