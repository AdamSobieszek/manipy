import pyperclip
import os
import re
import sys

def parse_and_create_files(clipboard_content):
    """
    Parses text containing file definitions and creates those files/directories.

    Assumes the format:
    1. path/to/file.py

    # path/to/file.py
    File content...
    
    2. another/file.py
    ...
    """

    # Regex to find the file path marker, capturing the path after the number
    # It looks for "**Number. `filepath`**"
    file_marker_regex = re.compile(r"^\*\*(\d+)\.\s+`(.+?)`\*\*$", re.MULTILINE)

    # Find all file markers in the content
    file_markers = list(file_marker_regex.finditer(clipboard_content))
    print(f"Found {len(file_markers)} potential file markers.")
    files_created = 0
    files_skipped = 0

    # Process each file marker
    for i, marker in enumerate(file_markers):
        # Get the file path from the match
        filepath = marker.group(2).strip()
        start_pos = marker.end()
        
        # Determine the end of this file's content (start of next file or end of text)
        end_pos = len(clipboard_content)
        if i < len(file_markers) - 1:
            end_pos = file_markers[i+1].start()
        
        # Extract the content between this marker and the next
        file_section = clipboard_content[start_pos:end_pos].strip()
        
        print(f"\nFound potential file: '{filepath}' at position {start_pos}")
        
        # Look for the file comment marker (# filepath)
        comment_marker = f"# {filepath}"
        comment_pos = file_section.find(comment_marker)
        
        if comment_pos != -1:
            # Extract content after the comment marker
            code_content = file_section[comment_pos + len(comment_marker):].strip()
            
            # Basic check if code content seems valid (not empty)
            if not code_content:
                print("   - Warning: Found file marker but code content is empty. Skipping.")
                files_skipped += 1
                continue
                
            # --- Create Directories and File ---
            try:
                # Get the directory part of the path
                directory = os.path.dirname(filepath)

                # Create directories if they don't exist
                if directory:
                    os.makedirs(directory, exist_ok=True)
                    print(f"   - Ensured directory exists: '{directory}'")

                # Write the file content (overwrite if exists)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(code_content)

                print(f"   - Successfully created file: '{filepath}'")
                files_created += 1

            except OSError as e:
                print(f"   - Error creating directory/file '{filepath}': {e}. Skipping.")
                files_skipped += 1
            except Exception as e:
                print(f"   - An unexpected error occurred for '{filepath}': {e}. Skipping.")
                files_skipped += 1
        else:
            print(f"   - Warning: Could not find comment marker '# {filepath}' in file section. Skipping.")
            files_skipped += 1

    print(f"\n--- Summary ---")
    print(f"Files successfully created: {files_created}")
    print(f"Segments skipped/failed: {files_skipped}")
    if files_created == 0 and len(file_markers) > 0:
        print("Warning: No files were created. Check clipboard content format.")

def main():
    print("Attempting to read from clipboard...")
    try:
        clipboard_content = pyperclip.paste().replace("\n---", "\n")
        if not clipboard_content or not isinstance(clipboard_content, str):
            print("Clipboard is empty or doesn't contain text.")
            sys.exit(1)

        print(f"Clipboard content loaded ({len(clipboard_content)} characters).")
        print("Parsing content and creating files...")
        parse_and_create_files(clipboard_content)

    except pyperclip.PyperclipException as e:
        print(f"Error accessing clipboard: {e}")
        print("Please ensure pyperclip is installed and works on your system.")
        print("You might need to install xclip or xsel on Linux.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()