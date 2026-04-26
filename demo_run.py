from framework.headless import execute_headless_new_run
from framework.services.new_run import NewRunRequest


def main() -> None:
    request = NewRunRequest(run_name="demo_run", notes="Headless demo run")
    summary = execute_headless_new_run(request)
    print(summary)


if __name__ == "__main__":
    main()
