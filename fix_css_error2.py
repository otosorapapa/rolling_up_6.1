# -*- coding: utf-8 -*-
import io

# Read the file with UTF-8 encoding
with io.open(r'c:\Users\10van\Dropbox\rolling_up_6.1\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("Total lines before:", len(lines))

# Delete lines from 1277 to 1759 (0-indexed: 1276 to 1758)
# Keep lines 0-1276 and lines from 1759 onwards
new_lines = lines[:1276] + lines[1759:]

print("Total lines after:", len(new_lines))

# Write back with UTF-8 encoding
with io.open(r'c:\Users\10van\Dropbox\rolling_up_6.1\app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File successfully updated!")
