import sys
import os

# Add the parent directory to sys.path so we can import app.py and src code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import main

if __name__ == '__main__':
    main()
