# Resolve a Git merge conflict
Category: Version control
Technologies: Git

Git reports a merge conflict when changes from two branches overlap. Run git status to list conflicted paths, open each file, and inspect the conflict markers (<<<<<<<, =======, >>>>>>>). Edit the file to retain the intended final content and remove all markers. Review the result, run the relevant checks, then stage resolved paths with git add and complete the merge with git commit. For rebase conflicts, use git rebase --continue after staging; use git rebase --abort to return to the pre-rebase state.

Do not blindly choose ours or theirs when both branches contain valid work. Use git diff and inspect surrounding context. Keep a copy of important uncommitted work before destructive operations.
