# Git LFS Setup for Video Files

## Current Status
✅ Git LFS is installed (version 3.7.0)
✅ `.gitattributes` file has been created with video file patterns
✅ Video file exists: `Screen Recording 2025-10-16 at 01.48.39.mov`
✅ README.md has been updated with proper video embed

## Commands to Run

### 1. Initialize Git LFS (if not already done)
```bash
git lfs install
```

### 2. Add the video file to Git LFS tracking
```bash
git lfs track "*.mov"
git lfs track "Screen Recording 2025-10-16 at 01.48.39.mov"
```

### 3. Add and commit the files
```bash
# Add the .gitattributes file
git add .gitattributes

# Add the video file (it will be tracked by LFS)
git add "Screen Recording 2025-10-16 at 01.48.39.mov"

# Add the updated README
git add README.md

# Commit the changes
git commit -m "Add demo video with Git LFS support

- Configure Git LFS for video files (.mov, .mp4, etc.)
- Add comprehensive demo video showing signup/login and RAG features
- Update README with proper video embed and description
- Set up .gitattributes for large file handling"
```

### 4. Push to GitHub
```bash
git push origin main
```

## Verification Commands

### Check LFS tracking
```bash
git lfs ls-files
```

### Check LFS status
```bash
git lfs status
```

### Verify file is tracked by LFS
```bash
git lfs ls-files | grep "Screen Recording"
```

## What Git LFS Does

1. **Stores large files externally**: The actual video file is stored on GitHub's LFS servers
2. **Keeps repository lightweight**: Only a pointer file is stored in the Git repository
3. **Seamless experience**: Users can still clone and work with the repository normally
4. **Automatic handling**: Git LFS automatically handles upload/download of large files

## File Size Information

```bash
# Check the video file size
ls -lh "Screen Recording 2025-10-16 at 01.48.39.mov"
```

## GitHub Display

Once pushed, the video will be displayed in the README on GitHub with:
- ✅ Proper video controls
- ✅ Fallback download link
- ✅ Responsive width (100%)
- ✅ Descriptive title and alt text

## Troubleshooting

If you encounter issues:

1. **LFS not installed**: Install with `brew install git-lfs` (macOS) or download from https://git-lfs.github.io/
2. **File already committed**: Use `git lfs migrate` to move existing files to LFS
3. **Push fails**: Ensure you have LFS quota available on GitHub

## Alternative: If Git LFS Quota is Limited

If you prefer not to use Git LFS due to quota concerns, you can:

1. Upload the video to a cloud service (YouTube, Vimeo, etc.)
2. Replace the video tag with an embedded iframe
3. Keep the video file locally and add it to `.gitignore`

Example for YouTube embed:
```html
<iframe width="100%" height="400" src="https://www.youtube.com/embed/VIDEO_ID" frameborder="0" allowfullscreen></iframe>
```
