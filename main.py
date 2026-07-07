import uvicorn

from aisimplerag.app import app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
