#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown to HTML Converter for RideBuddy Documentation
Creates professional HTML files that can be opened in Word and saved as DOC/DOCX
"""

import re
import os
from pathlib import Path
from datetime import datetime

def clean_text_for_html(text):
    """Clean and escape text for HTML format"""
    # Basic HTML escaping
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    return text

def convert_markdown_to_html(md_content, title="Document"):
    """Convert markdown content to professional HTML format"""
    
    html_content = []
    
    # HTML header with professional styling
    html_content.append('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>''' + clean_text_for_html(title) + '''</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background: white;
        }
        h1 {
            color: #2F5496;
            text-align: center;
            font-size: 28px;
            margin-bottom: 30px;
            border-bottom: 3px solid #2F5496;
            padding-bottom: 10px;
        }
        h2 {
            color: #1F3656;
            font-size: 22px;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #2F5496;
            padding-left: 15px;
        }
        h3 {
            color: #1F3656;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 12px;
        }
        h4 {
            color: #1F3656;
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        p {
            margin-bottom: 12px;
            text-align: justify;
        }
        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 90%;
        }
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            margin: 15px 0;
        }
        pre code {
            background: none;
            padding: 0;
        }
        ul, ol {
            margin-bottom: 15px;
            padding-left: 30px;
        }
        li {
            margin-bottom: 5px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #1F3656;
        }
        blockquote {
            border-left: 4px solid #2F5496;
            margin: 20px 0;
            padding: 10px 20px;
            background-color: #f9f9f9;
            font-style: italic;
        }
        hr {
            border: none;
            height: 2px;
            background-color: #ddd;
            margin: 30px 0;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 12px;
            color: #666;
        }
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .toc h2 {
            margin-top: 0;
            color: #2F5496;
            border: none;
            padding: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            margin-bottom: 8px;
        }
        .toc a {
            text-decoration: none;
            color: #1F3656;
        }
        .toc a:hover {
            color: #2F5496;
            text-decoration: underline;
        }
    </style>
</head>
<body>
''')
    
    # Add document title
    html_content.append(f'<h1>{clean_text_for_html(title)}</h1>')
    
    # Process markdown content
    lines = md_content.split('\n')
    in_code_block = False
    code_language = ""
    code_lines = []
    in_list = False
    list_type = None
    table_rows = []
    in_table = False
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Handle code blocks
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_language = line[3:].strip()
                code_lines = []
            else:
                in_code_block = False
                # Output code block
                if code_language:
                    html_content.append(f'<p><strong>Code ({clean_text_for_html(code_language)}):</strong></p>')
                html_content.append('<pre><code>')
                for code_line in code_lines:
                    html_content.append(clean_text_for_html(code_line))
                    html_content.append('\n')
                html_content.append('</code></pre>')
            i += 1
            continue
        
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
        
        # Handle tables
        if '|' in line and line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_rows = []
            
            # Skip separator lines
            if line.strip().replace('|', '').replace('-', '').replace(' ', '') == '':
                i += 1
                continue
            
            # Parse table row
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells:
                table_rows.append(cells)
            
            i += 1
            continue
        else:
            # End of table
            if in_table:
                in_table = False
                if table_rows:
                    html_content.append('<table>')
                    for row_idx, row in enumerate(table_rows):
                        if row_idx == 0:  # Header row
                            html_content.append('<tr>')
                            for cell in row:
                                html_content.append(f'<th>{clean_text_for_html(cell)}</th>')
                            html_content.append('</tr>')
                        else:  # Data row
                            html_content.append('<tr>')
                            for cell in row:
                                html_content.append(f'<td>{clean_text_for_html(cell)}</td>')
                            html_content.append('</tr>')
                    html_content.append('</table>')
        
        # Handle headings
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            level = len(heading_match.group(1))
            title_text = heading_match.group(2)
            # Remove emojis and clean title
            clean_title = re.sub(r'[^\w\s\-\(\)\[\]\.\,\:\;\/]', '', title_text).strip()
            
            if clean_title:
                # Create anchor for linking
                anchor = re.sub(r'[^\w\-]', '-', clean_title.lower()).strip('-')
                html_content.append(f'<h{level} id="{anchor}">{clean_text_for_html(clean_title)}</h{level}>')
            
            i += 1
            continue
        
        # Handle horizontal rules
        if line.strip() == '---':
            html_content.append('<hr>')
            i += 1
            continue
        
        # Handle lists
        if line.strip().startswith(('- ', '* ', '+ ')):
            if not in_list or list_type != 'ul':
                if in_list:
                    html_content.append(f'</{list_type}>')
                html_content.append('<ul>')
                in_list = True
                list_type = 'ul'
            
            text = line.strip()[2:]
            clean_text = clean_text_for_html(text)
            # Handle basic formatting in list items
            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', clean_text)
            clean_text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', clean_text)
            clean_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', clean_text)
            
            html_content.append(f'<li>{clean_text}</li>')
            
            i += 1
            continue
        
        # Handle numbered lists
        numbered_match = re.match(r'^\d+\.\s+(.+)$', line.strip())
        if numbered_match:
            if not in_list or list_type != 'ol':
                if in_list:
                    html_content.append(f'</{list_type}>')
                html_content.append('<ol>')
                in_list = True
                list_type = 'ol'
            
            text = numbered_match.group(1)
            clean_text = clean_text_for_html(text)
            # Handle basic formatting in list items
            clean_text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', clean_text)
            clean_text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', clean_text)
            clean_text = re.sub(r'`([^`]+)`', r'<code>\1</code>', clean_text)
            
            html_content.append(f'<li>{clean_text}</li>')
            
            i += 1
            continue
        else:
            # End of list
            if in_list:
                html_content.append(f'</{list_type}>')
                in_list = False
                list_type = None
        
        # Handle block quotes
        if line.strip().startswith('>'):
            quote_lines = []
            j = i
            while j < len(lines) and lines[j].strip().startswith('>'):
                quote_lines.append(lines[j].strip()[1:].strip())
                j += 1
            
            if quote_lines:
                quote_text = ' '.join(quote_lines)
                clean_quote = clean_text_for_html(quote_text)
                html_content.append(f'<blockquote><p>{clean_quote}</p></blockquote>')
                i = j
                continue
        
        # Handle regular paragraphs
        if line.strip():
            clean_line = clean_text_for_html(line)
            
            # Handle basic formatting
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', clean_line)
            clean_line = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', clean_line)
            clean_line = re.sub(r'`([^`]+)`', r'<code>\1</code>', clean_line)
            
            html_content.append(f'<p>{clean_line}</p>')
        else:
            # Empty line - just add some space
            html_content.append('<br>')
        
        i += 1
    
    # Close any open lists
    if in_list:
        html_content.append(f'</{list_type}>')
    
    # Add footer
    html_content.append(f'''
<div class="footer">
    <p>Generated on {datetime.now().strftime('%B %d, %Y')} - RideBuddy Pro Documentation</p>
    <p>Document can be opened in Microsoft Word and saved as DOC/DOCX format</p>
</div>
</body>
</html>''')
    
    return '\n'.join(html_content)

def convert_file_to_html(md_file, html_file):
    """Convert a markdown file to HTML format"""
    
    print(f"Converting {md_file} to {html_file}...")
    
    try:
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Extract title from filename
        title = Path(md_file).stem.replace('_', ' ').title()
        
        # Convert to HTML
        html_content = convert_markdown_to_html(md_content, title)
        
        # Write HTML file
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Successfully converted to {html_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {md_file}: {e}")
        return False

def main():
    """Main conversion function"""
    
    print("🔄 RideBuddy Documentation Converter - Markdown to HTML")
    print("=" * 60)
    print("Note: HTML files can be opened in Microsoft Word and saved as DOC/DOCX")
    print()
    
    # Define file pairs
    files_to_convert = [
        {
            'md': 'RIDEBUDDY_SYSTEM_DOCUMENTATION.md',
            'html': 'RideBuddy_System_Documentation.html'
        },
        {
            'md': 'INSTALLATION_AND_SETUP_GUIDE.md', 
            'html': 'RideBuddy_Installation_Setup_Guide.html'
        }
    ]
    
    successful_conversions = 0
    
    for file_pair in files_to_convert:
        md_path = file_pair['md']
        html_path = file_pair['html']
        
        if os.path.exists(md_path):
            if convert_file_to_html(md_path, html_path):
                successful_conversions += 1
        else:
            print(f"❌ Source file not found: {md_path}")
    
    print("\n" + "=" * 60)
    print(f"📊 Conversion Summary:")
    print(f"✅ Successful conversions: {successful_conversions}")
    print(f"📁 Total files processed: {len(files_to_convert)}")
    
    if successful_conversions == len(files_to_convert):
        print("🎉 All files converted successfully!")
    else:
        print("⚠️  Some conversions failed. Please check the error messages above.")
    
    print("\n📋 Generated HTML files:")
    for file_pair in files_to_convert:
        if os.path.exists(file_pair['html']):
            file_size = os.path.getsize(file_pair['html']) / 1024  # KB
            print(f"   • {file_pair['html']} ({file_size:.1f} KB)")
    
    print("\n💡 How to convert HTML to DOC/DOCX:")
    print("   1. Open Microsoft Word")
    print("   2. Go to File > Open")
    print("   3. Select the HTML file (change file type to 'All Files' if needed)")
    print("   4. Word will open the HTML file with formatting preserved")
    print("   5. Go to File > Save As")
    print("   6. Choose 'Word Document (*.docx)' or 'Word 97-2003 Document (*.doc)'")
    print("   7. Click Save")
    print("\n   Alternative: Right-click HTML file > Open with > Microsoft Word")

if __name__ == "__main__":
    main()