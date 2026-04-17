#!/usr/bin/env python3
"""
Launcher script for Banking Fraud Detection Web App
This script activates the virtual environment and starts the Flask application.
"""

import subprocess
import sys
import os

def main():
    """Launch the Flask web application."""
    print("🚀 Starting Banking Fraud Detection Web App...")
    print("=" * 50)

    # Check if we're on Windows
    if os.name == 'nt':
        # Windows PowerShell command
        cmd = [
            'powershell.exe',
            '-ExecutionPolicy', 'RemoteSigned',
            '-Command',
            '& .venv\\Scripts\\Activate.ps1 ; python app.py'
        ]

        try:
            print("📍 Opening web app at: http://127.0.0.1:5000")
            print("🌐 Web interface will open automatically in your browser")
            print("❌ Press Ctrl+C to stop the server")
            print()

            # Start the Flask app
            subprocess.run(cmd, cwd=os.getcwd())

        except KeyboardInterrupt:
            print("\n👋 Server stopped by user")
        except Exception as e:
            print(f"❌ Error starting server: {e}")

    else:
        print("❌ This launcher is designed for Windows. Please run manually:")
        print("python app.py")

if __name__ == "__main__":
    main()