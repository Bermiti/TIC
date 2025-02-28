#!/bin/bash
echo "Adding files to Git..."
git add .
git commit -m "Auto commit"
git push origin main
echo "✅ Push complete!"
