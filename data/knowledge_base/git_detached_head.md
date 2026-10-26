# Git Detached HEAD State and Commit Recovery
Category: Version Control & Source Code Management
Technologies: Git, Linux, Windows, macOS

When executing Git commands:
`You are in 'detached HEAD' state. You can look around, make experimental changes and commit them...`

## Root Cause
You checked out a specific commit hash, a remote tracking branch directly, or a tag (e.g. `git checkout v1.0.0` or `git checkout HEAD~1`) instead of checking out a local branch. New commits made in this state are not associated with any named branch and will be lost during garbage collection.

## Resolution Steps
1. Create and switch to a new branch containing your current commits:
   `git switch -c recovery-branch` or `git checkout -b recovery-branch`
2. If you already switched away from detached HEAD and lost your recent commits, inspect the reflog:
   `git reflog`
3. Locate the commit hash of your lost work, then check it out into a new branch:
   `git checkout -b rescued-work <commit-sha>`
4. Merge or rebase into your main branch:
   `git checkout main && git merge recovery-branch`
