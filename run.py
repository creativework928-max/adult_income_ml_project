import subprocess
import sys


def main():

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_pipeline",
        ],
        check=True
    )


if __name__ == "__main__":
    main()