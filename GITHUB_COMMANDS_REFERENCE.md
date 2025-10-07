# 🚀 RideBuddy Pro v2.1.0 - GitHub Quick Commands Reference

## Essential Git Commands for RideBuddy Repository

### 📋 Prerequisites Checklist
- [ ] Git installed on system
- [ ] GitHub account created
- [ ] Repository created on GitHub.com

---

## 🔧 Initial Setup Commands

### 1. Install Git (if not installed)
```powershell
# Option 1: Download from official website
# Go to: https://git-scm.com/download/windows

# Option 2: Install via Chocolatey
choco install git

# Option 3: Install via Winget
winget install Git.Git
```

### 2. Configure Git (One-time setup)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 3. Initialize Repository
```powershell
# Navigate to project directory
cd "C:\Users\ADX1KOR\TML\2ltr_PC\test"

# Initialize git repository
git init

# Add all files (respecting .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: RideBuddy Pro v2.1.0 - Complete AI Driver Monitoring System"
```

---

## 🌐 GitHub Connection Commands

### 1. Connect to GitHub Repository
```powershell
# Replace YOUR-USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR-USERNAME/ridebuddy-pro-v2.1.0.git

# Set main as default branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### 2. Verify Connection
```powershell
# Check remote connections
git remote -v

# Check repository status
git status
```

---

## 📦 GitHub Repository Creation Steps

### 1. Create Repository on GitHub
1. Go to: https://github.com/new
2. **Repository name**: `ridebuddy-pro-v2.1.0`
3. **Description**: 
   ```
   🚗 AI-Powered Driver Monitoring System - Advanced drowsiness and distraction detection with real-time processing (<50ms latency, 100% accuracy)
   ```
4. **Visibility**: Choose Public or Private
5. **Don't check**: "Add a README file" (we have existing files)
6. **Don't check**: "Add .gitignore" (we have custom .gitignore)
7. **Don't check**: "Choose a license" (add later if needed)
8. Click **"Create repository"**

---

## 🔄 Daily Git Workflow Commands

### Adding Changes
```powershell
# Check what's changed
git status

# Add specific files
git add filename.py

# Add all changes
git add .

# Add with interactive mode
git add -i
```

### Committing Changes
```powershell
# Commit with message
git commit -m "Add new feature: enhanced drowsiness detection"

# Commit all changes (tracked files)
git commit -am "Update documentation and fix bugs"
```

### Pushing to GitHub
```powershell
# Push to main branch
git push

# Force push (use carefully)
git push --force

# Push new branch
git push -u origin feature-branch-name
```

### Pulling from GitHub
```powershell
# Pull latest changes
git pull

# Pull with rebase
git pull --rebase

# Fetch without merging
git fetch
```

---

## 🌿 Branch Management

### Creating Branches
```powershell
# Create and switch to new branch
git checkout -b feature/new-algorithm

# Create branch without switching
git branch feature/new-algorithm

# Switch to existing branch
git checkout main
```

### Branch Operations
```powershell
# List all branches
git branch -a

# Delete local branch
git branch -d feature-name

# Delete remote branch
git push origin --delete feature-name
```

---

## 📁 Large File Management

### Git LFS (Large File Storage)
```powershell
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.zip"
git lfs track "*.pth"
git lfs track "*.onnx"

# Add .gitattributes
git add .gitattributes

# Push with LFS
git add large-file.zip
git commit -m "Add deployment package"
git push
```

### Alternative: GitHub Releases
1. Go to your repository on GitHub
2. Click **"Releases"** tab
3. Click **"Create a new release"**
4. Upload large files as **"Assets"**

---

## 🔍 Useful Git Commands

### Repository Information
```powershell
# View commit history
git log --oneline

# View file changes
git diff

# Check repository size
git count-objects -vH

# View remote repositories
git remote show origin
```

### Undoing Changes
```powershell
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Discard local changes
git checkout -- filename.py

# Revert specific commit
git revert commit-hash
```

---

## 🚨 Troubleshooting Common Issues

### Authentication Issues
```powershell
# Use Personal Access Token (recommended)
# Go to GitHub Settings > Developer settings > Personal access tokens
# Use token as password when prompted

# Set up SSH key (alternative)
ssh-keygen -t ed25519 -C "your.email@example.com"
# Add key to GitHub: Settings > SSH and GPG keys
```

### Large File Errors
```powershell
# Error: file exceeds GitHub's 100MB limit
# Solution 1: Use Git LFS
git lfs track "large-file.zip"
git add .gitattributes
git add large-file.zip

# Solution 2: Remove from history
git filter-branch --force --index-filter 'git rm --cached --ignore-unmatch large-file.zip'
```

### Merge Conflicts
```powershell
# View conflicts
git status

# Resolve conflicts manually in files, then:
git add resolved-file.py
git commit -m "Resolve merge conflict"
```

---

## 📊 Repository Maintenance

### Cleanup Commands
```powershell
# Clean untracked files
git clean -fd

# Optimize repository
git gc --aggressive

# Remove old branches
git remote prune origin
```

### Backup Commands
```powershell
# Clone repository
git clone https://github.com/YOUR-USERNAME/ridebuddy-pro-v2.1.0.git

# Create archive
git archive --format=zip --output=backup.zip main
```

---

## 🎯 RideBuddy Specific Workflow

### Development Workflow
```powershell
# 1. Create feature branch
git checkout -b feature/improve-accuracy

# 2. Make changes to code
# Edit files: ridebuddy_optimized_gui.py, etc.

# 3. Test changes
python ridebuddy_optimized_gui.py

# 4. Commit changes
git add .
git commit -m "Improve drowsiness detection accuracy by 2%"

# 5. Push feature branch
git push -u origin feature/improve-accuracy

# 6. Create Pull Request on GitHub
# Go to repository > Pull requests > New pull request

# 7. Merge after review
# Delete feature branch after merge
git checkout main
git pull
git branch -d feature/improve-accuracy
```

### Release Workflow
```powershell
# 1. Create release branch
git checkout -b release/v2.2.0

# 2. Update version numbers
# Edit version in files

# 3. Create deployment packages
python create_production_package.py
python create_developer_package.py

# 4. Commit release
git add .
git commit -m "Release v2.2.0: Enhanced AI accuracy and performance"

# 5. Merge to main
git checkout main
git merge release/v2.2.0

# 6. Tag release
git tag -a v2.2.0 -m "RideBuddy Pro v2.2.0"
git push origin v2.2.0

# 7. Create GitHub Release
# Upload deployment packages as release assets
```

---

## 📞 Quick Help

### Get Help
```powershell
# General git help
git help

# Specific command help
git help commit
git help push

# Git version
git --version
```

### Useful Aliases
```powershell
# Set up helpful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.cm commit
git config --global alias.pl pull
git config --global alias.ps push
```

---

## 🎉 Success Checklist

After completing GitHub setup:

- [ ] Repository accessible at: `https://github.com/YOUR-USERNAME/ridebuddy-pro-v2.1.0`
- [ ] All important files uploaded (excluding large files)
- [ ] .gitignore working correctly
- [ ] Repository description and topics added
- [ ] Large deployment packages uploaded as releases
- [ ] Documentation accessible and properly formatted
- [ ] Clone test successful from different location

---

**🚀 Your RideBuddy Pro repository is now live on GitHub!**

Share the repository link with:
- Management team for review
- Development collaborators  
- Potential users and contributors
- Technical stakeholders

*GitHub Commands Reference - October 6, 2025*  
*RideBuddy Pro v2.1.0 - Complete Repository Management*