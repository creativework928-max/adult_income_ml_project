import requests


def main():

    url = (
        "http://localhost:8000/health"
    )

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    print(
        "API:",
        response.json()
    )


if __name__ == "__main__":
    main()