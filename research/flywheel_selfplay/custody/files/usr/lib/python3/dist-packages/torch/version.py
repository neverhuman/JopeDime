from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.7.0'
debug = False
cuda: Optional[str] = '12.8'
git_version = 'Unknown'
hip: Optional[str] = None
xpu: Optional[str] = None
