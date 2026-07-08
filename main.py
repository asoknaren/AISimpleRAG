import uvicorn

from aisimplerag.app import app


def main() -> None:
    uvicorn.run(app, host="localhost", port=8765)


if __name__ == "__main__":
    main()
