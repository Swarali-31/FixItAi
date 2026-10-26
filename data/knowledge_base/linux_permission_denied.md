# Linux Permission Denied & EACCES File Access Errors
Category: Operating Systems & Filesystem Permissions
Technologies: Linux, Ubuntu, macOS, Bash, Git, Python

When attempting to execute a script, write to a log file, or bind to low-numbered network ports (<1024):
`Permission denied` or `errno: -13, code: 'EACCES'` or `zsh: permission denied: ./run.sh`.

## Root Cause
1. The script or binary lacks execution bit permissions (`+x`).
2. The user account lacks ownership or write permissions over the destination folder.
3. The process attempted to bind to a privileged port (<1024) without `sudo` or capability permissions (`CAP_NET_BIND_SERVICE`).

## Resolution Steps
1. Add execution permissions to the script:
   `chmod +x filename.sh`
2. Change file or directory ownership to your active user account:
   `sudo chown -R $USER:$USER /path/to/project`
3. For network services, use high unprivileged ports (e.g. 8000, 8080, 5000) instead of ports 80 or 443.
