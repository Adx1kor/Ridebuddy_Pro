#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Markdown to Rich Text Format (RTF) Converter for RideBuddy Documentation
Creates RTF files that can be opened by Microsoft Word and other word processors
"""

import re
import os
from pathlib import Path
from datetime import datetime

def clean_text_for_rtf(text):
    """Clean and escape text for RTF format"""
    # Remove emojis and special Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Escape RTF special characters
    text = text.replace('\\', '\\\\')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text

def convert_markdown_to_rtf(md_content, title="Document"):
    """Convert markdown content to RTF format"""
    
    rtf_content = []
    
    # RTF header
    rtf_content.append(r"{\rtf1\ansi\deff0")
    
    # Font table
    rtf_content.append(r"{\fonttbl")
    rtf_content.append(r"{\f0\froman\fcharset0 Times New Roman;}")
    rtf_content.append(r"{\f1\fswiss\fcharset0 Arial;}")
    rtf_content.append(r"{\f2\fmodern\fcharset0 Courier New;}")
    rtf_content.append(r"}")
    
    # Color table
    rtf_content.append(r"{\colortbl;")
    rtf_content.append(r"\red0\green0\blue0;")      # Black
    rtf_content.append(r"\red47\green84\blue150;")  # Professional Blue
    rtf_content.append(r"\red31\green54\blue86;")   # Dark Blue
    rtf_content.append(r"\red204\green204\blue204;") # Light Gray
    rtf_content.append(r"}")
    
    # Document title
    rtf_content.append(r"\f1\fs32\b\qc\cf2")  # Arial, 16pt, Bold, Centered, Blue
    rtf_content.append(clean_text_for_rtf(title))
    rtf_content.append(r"\par\par")
    
    # Reset formatting
    rtf_content.append(r"\f0\fs22\b0\ql\cf1\par")  # Times New Roman, 11pt, Normal, Left, Black
    
    # Process markdown content line by line
    lines = md_content.split('\n')
    in_code_block = False
    code_language = ""
    
    for line in lines:
        line = line.rstrip()
        
        # Handle code blocks
        if line.startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_language = line[3:].strip()
                if code_language:
                    rtf_content.append(rf"\b Code ({clean_text_for_rtf(code_language)}):\b0\par")
                rtf_content.append(r"\f2\fs18")  # Courier New, 9pt
            else:
                in_code_block = False
                rtf_content.append(r"\f0\fs22\par")  # Back to normal font
            continue
        
        if in_code_block:
            rtf_content.append(clean_text_for_rtf(line))
            rtf_content.append(r"\par")
            continue
        
        # Handle headings
        if line.startswith('#'):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2)
                clean_title = clean_text_for_rtf(title_text)
                
                if level == 1:
                    rtf_content.append(r"\f1\fs28\b\cf3")  # Arial, 14pt, Bold, Dark Blue
                elif level == 2:
                    rtf_content.append(r"\f1\fs24\b\cf3")  # Arial, 12pt, Bold, Dark Blue
                elif level == 3:
                    rtf_content.append(r"\f1\fs22\b\cf3")  # Arial, 11pt, Bold, Dark Blue
                else:
                    rtf_content.append(r"\f1\fs20\b\cf3")  # Arial, 10pt, Bold, Dark Blue
                
                rtf_content.append(clean_title)
                rtf_content.append(r"\par\f0\fs22\b0\cf1")  # Reset to normal
                continue
        
        # Handle horizontal rules
        if line.strip() == '---':
            rtf_content.append(r"\brdrb\brdrs\brdrw10\brsp20")
            rtf_content.append(r"\par\par")
            continue
        
        # Handle bullet points
        if line.strip().startswith(('- ', '* ', '+ ')):
            text = line.strip()[2:]
            clean_text = clean_text_for_rtf(text)
            rtf_content.append(rf"\bullet\tab {clean_text}\par")
            continue
        
        # Handle numbered lists
        numbered_match = re.match(r'^\d+\.\s+(.+)$', line.strip())
        if numbered_match:
            text = numbered_match.group(1)
            clean_text = clean_text_for_rtf(text)
            rtf_content.append(rf"{clean_text}\par")
            continue
        
        # Handle tables (simplified)
        if '|' in line and line.strip().startswith('|'):
            # Convert table row to simple text
            cells = [cell.strip() for cell in line.split('|')[1:-1]]
            if cells and not all(cell.replace('-', '').replace(' ', '') == '' for cell in cells):
                row_text = ' | '.join(clean_text_for_rtf(cell) for cell in cells)
                rtf_content.append(rf"{row_text}\par")
            continue
        
        # Handle block quotes
        if line.strip().startswith('>'):
            quote_text = line.strip()[1:].strip()
            clean_quote = clean_text_for_rtf(quote_text)
            rtf_content.append(rf"\li720\ri720\i {clean_quote}\i0\li0\ri0\par")
            continue
        
        # Handle regular paragraphs
        if line.strip():
            clean_line = clean_text_for_rtf(line)
            
            # Handle basic formatting
            # Bold text (**text**)
            clean_line = re.sub(r'\*\*([^*]+)\*\*', r'\\b \1\\b0 ', clean_line)
            
            # Italic text (*text*)
            clean_line = re.sub(r'\*([^*]+)\*', r'\\i \1\\i0 ', clean_line)
            
            # Inline code (`text`)
            clean_line = re.sub(r'`([^`]+)`', r'\\f2 \1\\f0 ', clean_line)
            
            rtf_content.append(clean_line)
            rtf_content.append(r"\par")
        else:
            rtf_content.append(r"\par")
    
    # Add footer
    rtf_content.append(r"\par\par")
    rtf_content.append(r"\fs16\i")
    rtf_content.append(f"Generated on {datetime.now().strftime('%B %d, %Y')} - RideBuddy Pro Documentation")
    rtf_content.append(r"\i0")
    
    # RTF footer
    rtf_content.append("}")
    
    return ''.join(rtf_content)

def convert_file_to_rtf(md_file, rtf_file):
    """Convert a markdown file to RTF format"""
    
    print(f"Converting {md_file} to {rtf_file}...")
    
    try:
        # Read markdown file
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Extract title from filename
        title = Path(md_file).stem.replace('_', ' ').title()
        
        # Convert to RTF
        rtf_content = convert_markdown_to_rtf(md_content, title)
        
        # Write RTF file
        with open(rtf_file, 'w', encoding='utf-8') as f:
            f.write(rtf_content)
        
        print(f"✅ Successfully converted to {rtf_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {md_file}: {e}")
        return False

def main():
    """Main conversion function"""
    
    print("🔄 RideBuddy Documentation Converter - Markdown to RTF")
    print("=" * 60)
    print("Note: RTF files can be opened and converted to DOC/DOCX in Microsoft Word")
    print()
    
    # Define file pairs
    files_to_convert = [
        {
            'md': 'RIDEBUDDY_SYSTEM_DOCUMENTATION.md',
            'rtf': 'RideBuddy_System_Documentation.rtf'
        },
        {
            'md': 'INSTALLATION_AND_SETUP_GUIDE.md', 
            'rtf': 'RideBuddy_Installation_Setup_Guide.rtf'
        }
    ]
    
    successful_conversions = 0
    
    for file_pair in files_to_convert:
        md_path = file_pair['md']
        rtf_path = file_pair['rtf']
        
        if os.path.exists(md_path):
            if convert_file_to_rtf(md_path, rtf_path):
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
    
    print("\n📋 Generated RTF files:")
    for file_pair in files_to_convert:
        if os.path.exists(file_pair['rtf']):
            file_size = os.path.getsize(file_pair['rtf']) / 1024  # KB
            print(f"   • {file_pair['rtf']} ({file_size:.1f} KB)")
    
    print("\n💡 How to convert RTF to DOC/DOCX:")
    print("   1. Open the RTF file in Microsoft Word")
    print("   2. Go to File > Save As")
    print("   3. Choose 'Word Document (*.docx)' or 'Word 97-2003 Document (*.doc)'")
    print("   4. Click Save")

if __name__ == "__main__":
    main()