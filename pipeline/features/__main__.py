import sys

from pipeline.ball.stage import main
from pipeline.features import run

if __name__ == "__main__":
    sys.exit(main("features", run, needs="events.parquet"))
