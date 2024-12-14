import subprocess
import os

# Define the commands to run
commands = [

]

# Log file path
log_file = "execution_log.txt"

# Open the log file for writing
with open(log_file, "w") as log:
    for i, command in enumerate(commands, start=1):
        log.write(f"Running command {i}: {command}\n")
        log.write("=" * 80 + "\n")
        
        try:
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            # Log the output
            log.write("Output:\n")
            log.write(result.stdout + "\n")
            
            # Log any errors
            if result.stderr:
                log.write("Errors, warnings, messages:\n")
                log.write(result.stderr + "\n")
            
            # Log the exit code
            log.write(f"Exit code: {result.returncode}\n")
            
        except Exception as e:
            # Log any exceptions raised during execution
            log.write(f"Exception occurred while running command {i}: {e}\n")
        
        log.write("=" * 80 + "\n\n")

print(f"Execution completed. Logs are saved in {os.path.abspath(log_file)}")