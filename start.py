#!/usr/bin/env python3
"""
Simple startup script for Karpathy agent.
Runs sandbox setup and then starts the ADK web interface.
"""
import subprocess
import sys
from karpathy.utils import setup_sandbox


def main():
    """Run setup and start the ADK web interface."""
    print("Starting Karpathy agent...")
    
    # Step 1: Setup sandbox
    print("\nSetting up sandbox...")
    try:
        setup_sandbox()
    except Exception as e:
        print(f"Error setting up sandbox: {e}")
        sys.exit(1)
    
    # Step 2: Start ADK web interface
    print("\nStarting ADK web interface...")
    try:
        subprocess.run(["adk", "web"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting ADK web: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

