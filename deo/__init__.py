"""
deo — Terminal AI coding agent powered by Deep Agents + ChromaDB.

Public API for programmatic use:
  from deo import DeoConfig, create_deo_agent
  config = DeoConfig.load("/path/to/project")
"""

from deo.config import DeoConfig
from deo.agents.factory import create_deo_agent
from deo.memory.chroma_manager import ChromaManager
from deo.memory.indexer import CodebaseIndexer

__version__ = "0.1.0"
__all__ = ["DeoConfig", "create_deo_agent", "ChromaManager", "CodebaseIndexer"]
