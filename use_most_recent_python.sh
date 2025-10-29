#!/bin/bash
# Keith Briggs 2025-09-03

if command -v python3.13 >/dev/null 2>&1; then 
  #echo "using python3.13..."
  python3.13 "$@"
elif command -v python3.11 >/dev/null 2>&1; then 
  #echo "using python3.11..."
  python3.11 "$@"
else 
  python3 "$@"; 
fi
