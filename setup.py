import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'scikit-learn',
        'matplotlib',
        'plotly'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def create_users_file():
    """Create empty users file if it doesn't exist"""
    import json
    import os
    
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump({}, f)
        print("✅ Created users.json file")
    else:
        print("✅ users.json already exists")

if __name__ == "__main__":
    print("🚀 Setting up Genomic Advisor...")
    install_packages()
    create_users_file()
    print("\n🎉 Setup complete! Run: streamlit run genomics_app.py")