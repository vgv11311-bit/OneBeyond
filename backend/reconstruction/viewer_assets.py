import os
from pathlib import Path
def renderer_root():
    return Path(os.environ.get('ONE_BEYOND_SPARK_DIST', str(Path(__file__).resolve().parents[2]/'frontend/node_modules/@sparkjsdev/spark/dist')))
