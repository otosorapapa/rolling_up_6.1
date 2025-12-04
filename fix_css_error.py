import sys

# Read the file
with open(r'c:\Users\10van\Dropbox\rolling_up_6.1\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines before: {len(lines)}")

# Find the line with "</style>" after line 1020
style_close_line = -1
for i in range(1020, len(lines)):
    if '</style>' in lines[i]:
        style_close_line = i
        break

print(f"First </style> found at line: {style_close_line + 1}")

# Find the second "</style>" 
second_style_close = -1
for i in range(style_close_line + 1, len(lines)):
    if '</style>' in lines[i]:
        second_style_close = i
        break

print(f"Second </style> found at line: {second_style_close + 1}")

# Remove lines between first </style> and second </style>
if style_close_line != -1 and second_style_close != -1:
    # Keep lines up to and including first </style> + the closing of st.markdown
    # Then skip to brand_override_template line
    new_lines = lines[:style_close_line + 4]  # Include </style>, """, unsafe_allow_html=True, )
    
    # Find the line with "brand_override_template"
    brand_line = -1
    for i in range(second_style_close, len(lines)):
        if 'brand_override_template = Template(' in lines[i]:
            brand_line = i
            break
    
    print(f"brand_override_template found at line: {brand_line + 1}")
    
    if brand_line != -1:
        new_lines.extend(lines[brand_line:])
        
        print(f"Total lines after: {len(new_lines)}")
        
        # Write back
        with open(r'c:\Users\10van\Dropbox\rolling_up_6.1\app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print("File successfully updated!")
    else:
        print("ERROR: Could not find brand_override_template line")
else:
    print("ERROR: Could not find style tags")
