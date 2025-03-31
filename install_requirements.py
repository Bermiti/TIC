#install_requirements.py
import subprocess
import sys

def install_requirements():
    """
    Installs all the required libraries for the portfolio analysis.
    """
    libraries = [
        "pandas",
        "openpyxl",
        "numpy",
        "yfinance",
        "tqdm",
        "xlrd",# for reading older Excel files if needed
        "os" 
    ]
    for lib in libraries:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

if __name__ == "__main__":
    install_requirements()
