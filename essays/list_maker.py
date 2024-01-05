import os
from datetime import datetime

OUTPUT = 'README.md'
EXCLUDE_DIRS = ['images']
ROOT_DIR = '.'

def format_title(filename):
    # Remove the file extension and replace dashes with spaces
    name_without_ext = os.path.splitext(filename)[0].replace('-', ' ')
    # Capitalize the first letter of each word
    return name_without_ext.title()

def generate_markdown_list(dir_path):
    sections = {}
    for root, dirs, files in os.walk(dir_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]  # Exclude specified directories
        section = os.path.relpath(root, dir_path)
        if section == '.':
            continue  # Skip the root directory itself for section listing
        section = section.replace('_', ' ').title()  # Format the section heading
        essays = []
        for file in files:
            if not file.startswith('.') and file.endswith('.md'):  # Ignore hidden and non-markdown files
                formatted_title = format_title(file)
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, dir_path)
                essay_str = f"- [{formatted_title}]({relative_path})"
                essays.append(essay_str)
        if essays:
            essays.sort()  # Sort the essays within their section
            sections[section] = essays  # Store the sorted essays in the dictionary under their section

    markdown_list = []
    for section in sorted(sections.keys()):  # Sort the sections alphabetically before adding to markdown
        section_str = f"## {section}\n"
        markdown_list.append(section_str)
        print('\n', section_str)
        print('\n'.join(sections[section]))
        markdown_list.extend(sections[section])  # Add sorted essays
        markdown_list.append('\n')  # Add newline after each section for spacing

    return '\n'.join(markdown_list)  # Join all markdown entries

print(f"Making a list of essays in {OUTPUT}...")

with open(OUTPUT, 'w') as f:
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(f"Auto-generated list of essays: {date_time}\n\n")
    f.write("# CWK Essays\n\n")  # Add two newlines for spacing after the title
    markdowns = generate_markdown_list(ROOT_DIR)
    f.write(markdowns)  # Write the markdown content