import subprocess
import argparse

def run_server():
    """Function to run the Django server."""
    subprocess.run(["python", "manage.py", "runserver"])

def create_project(project_name):
    """Function to create a new Django project."""
    subprocess.run(["python", "manage.py ", "startapp", project_name])

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Manage Django server and projects.")
    parser.add_argument(
        "command",
        help="Command to execute: 'run' to start the server or 'create_project=<project_name>' to create a new project.",
        type=str
    )
    
    args = parser.parse_args()

    # Check the command provided
    if args.command == "run":
        run_server()
    elif args.command.startswith("create_project="):
        project_name = args.command.split("=")[1]
        create_project(project_name)
    else:
        print("Invalid command. Use 'run' to start the server or 'create_project=<project_name>' to create a new project.")
