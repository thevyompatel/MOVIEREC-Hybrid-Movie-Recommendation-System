import os
import glob

files = glob.glob('src/*.py') + ['app.py', 'app/streamlit_app.py', 'verify.py']
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            
        new_content = content.replace('\\"', '"')
        
        with open(f, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Fixed {f}")
    except Exception as e:
        print(f"Failed on {f}: {e}")
