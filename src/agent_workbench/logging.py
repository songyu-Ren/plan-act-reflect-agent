import logging
import sys
from pathlib import Path
from typing import Optional

from agent_workbench.settings import Settings


def setup_logging(settings: Settings, session_id: Optional[str] = None) -> logging.Logger:
    log_dir = Path(settings.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "agent.log"
    if session_id:
        log_file = log_dir / f"agent_{session_id}.log"
    
    logging.basicConfig(
        level=getattr(logging, settings.monitoring.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("agent_workbench")