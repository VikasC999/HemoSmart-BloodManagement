"""
HemoSmart evaluation scripts. Importing this package points DATABASE_URL
at a throwaway SQLite file first, so no evaluation can ever touch the
real database -- even one that only imports a module which happens to
initialise db.database.
"""

import os
import tempfile

_DB_DIR = tempfile.mkdtemp(prefix="hemosmart-eval-")
os.environ["DATABASE_URL"] = f"sqlite:///{os.path.join(_DB_DIR, 'eval.db')}"
os.environ.setdefault("JWT_SECRET_KEY", "evaluation")
