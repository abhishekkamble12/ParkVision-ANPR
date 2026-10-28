## Run these commands in PowerShell or Windows Terminal, one by one.
## The whole thing takes ~2 minutes.

# Step 1 — Go into the project folder
cd d:\learning_cv\no_plate_proj

# Step 2 — Initialize git (skip if already done)
git init

# Step 3 — Configure your name/email if this is a fresh machine
git config user.name "Abhishek Kamble"
git config user.email "your-email@example.com"

# Step 4 — Stage all files
git add .

# Step 5 — First commit
git commit -m "Initial commit: ParkVision ANPR - YOLOv8 + EasyOCR + FastAPI + Streamlit + Docker"

# Step 6 — Create GitHub repo and push in ONE command (requires gh CLI logged in)
gh repo create ParkVision-ANPR `
  --public `
  --description "Indian Smart Parking & ANPR System | YOLOv8 · EasyOCR · FastAPI · Streamlit · Docker" `
  --source=. `
  --remote=origin `
  --push

## ✅ Done! GitHub will print your repo URL.
## It will be: https://github.com/YOUR_USERNAME/ParkVision-ANPR
