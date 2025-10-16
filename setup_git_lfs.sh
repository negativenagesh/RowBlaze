#!/bin/bash

# Git LFS Setup Script for RowBlaze Video Demo
# This script sets up Git LFS for the demo video and commits it properly

set -e  # Exit on any error

echo "🚀 Setting up Git LFS for RowBlaze demo video..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}❌ Git LFS is not installed. Please install it first:${NC}"
    echo "   macOS: brew install git-lfs"
    echo "   Other: https://git-lfs.github.io/"
    exit 1
fi

echo -e "${GREEN}✅ Git LFS is installed ($(git lfs version))${NC}"

# Initialize Git LFS if not already done
echo -e "${BLUE}🔧 Initializing Git LFS...${NC}"
git lfs install

# Check if video file exists
VIDEO_FILE="Screen Recording 2025-10-16 at 01.48.39.mov"
if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}❌ Video file not found: $VIDEO_FILE${NC}"
    exit 1
fi

# Get file size
FILE_SIZE=$(ls -lh "$VIDEO_FILE" | awk '{print $5}')
echo -e "${BLUE}📹 Video file size: $FILE_SIZE${NC}"

# Track video files with LFS
echo -e "${BLUE}📝 Configuring LFS tracking for video files...${NC}"
git lfs track "*.mov"
git lfs track "*.mp4"
git lfs track "*.avi"
git lfs track "*.mkv"
git lfs track "*.webm"

# Verify .gitattributes was created/updated
if [ -f ".gitattributes" ]; then
    echo -e "${GREEN}✅ .gitattributes file configured${NC}"
else
    echo -e "${RED}❌ .gitattributes file not found${NC}"
    exit 1
fi

# Check if files are already staged/committed
if git ls-files --error-unmatch "$VIDEO_FILE" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Video file is already tracked by Git${NC}"
    echo -e "${BLUE}🔄 Migrating to LFS...${NC}"
    git lfs migrate import --include="*.mov" --everything
else
    echo -e "${BLUE}➕ Adding new files...${NC}"
fi

# Add files to staging
echo -e "${BLUE}📦 Staging files...${NC}"
git add .gitattributes
git add "$VIDEO_FILE"
git add README.md
git add GIT_LFS_SETUP.md
git add setup_git_lfs.sh

# Check LFS status
echo -e "${BLUE}🔍 Checking LFS status...${NC}"
git lfs status

# Show what will be committed
echo -e "${BLUE}📋 Files to be committed:${NC}"
git status --porcelain

# Commit with descriptive message
echo -e "${BLUE}💾 Committing changes...${NC}"
git commit -m "Add demo video with Git LFS support

- Configure Git LFS for video files (.mov, .mp4, etc.)
- Add comprehensive demo video (18MB) showing signup/login and RAG features
- Update README with proper video embed and description
- Set up .gitattributes for large file handling
- Add Git LFS setup documentation and automation script

The video demonstrates:
- User authentication (signup/login)
- Document upload and processing
- Normal RAG vs Agentic RAG comparison
- Complete user workflow and features"

echo -e "${GREEN}✅ Files committed successfully!${NC}"

# Show LFS files
echo -e "${BLUE}📁 LFS tracked files:${NC}"
git lfs ls-files

echo -e "${GREEN}🎉 Git LFS setup complete!${NC}"
echo -e "${YELLOW}📤 Next step: Push to GitHub with 'git push origin main'${NC}"

# Provide push instructions
echo -e "${BLUE}📋 To push to GitHub:${NC}"
echo "   git push origin main"
echo ""
echo -e "${BLUE}🌐 The video will be displayed on GitHub at:${NC}"
echo "   https://github.com/your-username/RowBlaze#-demo-video"
echo ""
echo -e "${BLUE}💡 Note: GitHub LFS has usage limits. Current file size: $FILE_SIZE${NC}"
