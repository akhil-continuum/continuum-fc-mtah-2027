import sys

from pipeline.ball import run
from pipeline.ball.stage import main

if __name__ == "__main__":
    sys.exit(main("ball", run, needs="ball_raw.parquet"))
