"""
Common classes and utilities for the streaming pipeline
"""

from datetime import datetime
from typing import Any


class PipelineStatus:
    """Status object for pipeline updates"""
    def __init__(self, stage: str, message: str, progress: float = 0.0, data: Any = None, error: str = None):
        self.stage = stage
        self.message = message
        self.progress = progress  # 0.0 to 1.0
        self.data = data
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'stage': self.stage,
            'message': self.message,
            'progress': self.progress,
            'data': self.data,
            'error': self.error,
            'timestamp': self.timestamp
        } 