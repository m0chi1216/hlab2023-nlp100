"""
The file enwiki-country.json.gz stores Wikipedia articles in the format:

    -Each line stores a Wikipedia article in JSON format
    -Each JSON document has key-value pairs:
        --Title of the article as the value for the title key
        --Body of the article as the value for the text key
    -The entire file is compressed by gzip

Write codes that perform the following jobs.
"""

import gzip
import json
import re
import urllib.request

# 26. Function to remove emphasis MediaWiki markups from the field values
def remove_emphasis_markup(value):
    # Remove emphasis markup by replacing '' characters with an empty string
    return re.sub(r"''+", "", value)

# 27. Function to remove internal link markup from the field values and convert them to plain text
def remove_internal_links(value):
    # Remove internal link markup by replacing [[link]] with link text
    value = re.sub(r'\[\[([^|\]]+\|)?([^|\]]+)\]\]', r'\2', value)
    
    return value

# 28. Function to remove MediaWiki markups from the field values and convert them to plain text
def remove_mediawiki_markups(value):
    # Remove emphasis markup by replacing '' characters with an empty string
    value = re.sub(r"''+", "", value)
    
    # Remove bold markup by replacing ''' characters with an empty string
    value = re.sub(r"'''", "", value)
    
    # Remove italic markup by replacing ''''' characters with an empty string
    value = re.sub(r"'''''", "", value)
    
    # Remove internal link markup by replacing [[link]] with link text
    value = re.sub(r'\[\[([^|\]]+\|)?([^|\]]+)\]\]', r'\2', value)
    
    # Remove other MediaWiki markups by replacing {{...}} and {|...|} with an empty string
    value = re.sub(r'\{\{.*?\}\}|\{\|.*?\|\}', "", value, flags=re.DOTALL)
    
    return value.strip()

# 29. Function to extract the URL of the flag image from the Infobox
def extract_flag_url(infobox_text):
    # Extract the file name of the flag image
    flag_match = re.search(r'\|flag\s*=\s*(.+)', infobox_text, re.IGNORECASE)
    if flag_match:
        flag_filename = flag_match.group(1)
        
        # Make a request to MediaWiki API to get information about the file
        api_url = 'https://en.wikipedia.org/w/api.php'
        params = {
            'action': 'query',
            'prop': 'imageinfo',
            'iiprop': 'url',
            'titles': f'File:{flag_filename}',
            'format': 'json'
        }
        
        response = urllib.request.get(api_url, params=params)
        data = response.json()
        
        # Extract the URL of the flag image from the API response
        pages = data.get('query', {}).get('pages', {})
        for page in pages.values():
            imageinfo = page.get('imageinfo', [])
            if imageinfo:
                flag_url = imageinfo[0].get('url')
                return flag_url

    return None


# define the path of the file
file_path = '.enwiki-country.json.gz'

# decompress
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    # read the file
    for line in f:
        data = json.loads(line)

        # 20.Read JSON documents
        # Read the JSON documents and output the body of the article about the United Kingdom.
        # Reuse the output in problems 21-29.
        if data['title'] == 'United Kingdom':
            # output
            text = data['text']
            with open('.UnitedKingdom.json', 'w') as f:
                json.dump(text, f)

            # 21.Lines with category names
            # Extract lines that define the categories of the article.
            lines = text.split('\n')
            category_lines = [line for line in lines if line.startswith('[[Category:')]

            # print the extracted categories
            print('\n"21.Extracted Categories"\n')
            for category_line in category_lines:
                print(category_line)


            # 22.Category names
            # Extract the category names of the article.
            category_names = [re.search(r'\[\[Category:(.*?)\]\]', line).group(1) for line in category_lines]

            # print the category names
            print('\n"22.Category Names"\n')
            for category_name in category_names:
                print(category_name)


            # 23.Section structure
            # Extract section names in the article with their levels.
            # For example, the level of the section is 1 for the MediaWiki markup "== Section name ==".
            sections = re.findall(r'(={2,})\s*(.*?)\s*(={2,})', text)

            # Print the section names with their levels
            print('\n"23.Section Names"\n')
            for section in sections:
                level = len(section[0]) - 1  # Calculate the level based on the number of "="
                section_name = section[1]
                print(f"Level {level}: {section_name}")


            # 24.Media references
            # Extract references to media files linked from the article.
            media_files = re.findall(r'\[\[File:(.*?)\]\]', text)

            # Print the references to media files
            print('\n"24.Media References"\n')
            for media_file in media_files:
                print(media_file)


            # 25.Infobox ?
            # Extract field names and their values in the Infobox “country”, and store them in a dictionary object.
            infobox_match = re.search(r'{{Infobox country(.*?)}}', text, re.DOTALL)
            if infobox_match:
                # infobox_match: re.match   infobox_text: str
                infobox_text = infobox_match.group(1)

                # Extract field names and values using regex(正则匹配)
                fields = re.findall(r'\|([\w\s]+)\s*=\s*(.*?)\n', infobox_text)

                # Store field names and values in a dictionary
                infobox_dict = dict(fields)

                print(infobox_dict)


                # 26.Remove emphasis markups
                # In addition to the process of the problem 25, remove emphasis MediaWiki markups from the values
                print('\n"26.Field name and Value"\n')
                for field in fields:
                    field_name = field[0]
                    field_value = field[1]

                    # Remove emphasis markup from the field value
                    field_value = remove_emphasis_markup(field_value)

                    # Print the field name and updated value
                    print(f"{field_name}: {field_value}")


                # 27. Remove internal links
                # In addition to the process of the problem 26, remove internal links from the values.
                print('\n"27.Field name and Value"\n')
                for field in fields:
                    field_name = field[0]
                    field_value = field[1]

                    field_value = remove_internal_links(field_value)
                    
                    # Print the field name and updated value
                    print(f"{field_name}: {field_value}")
    

                # 28. Remove MediaWiki markups
                # In addition to the process of the problem 27, remove MediaWiki markups from the values as much as you can,
                # and obtain the basic information of the country in plain text format.
                print('\n"28.Field name and Value"\n')
                for field in fields:
                    field_name = field[0]
                    field_value = field[1]

                    # Remove MediaWiki markups and convert to plain text
                    field_value = remove_mediawiki_markups(field_value)
                    
                    # Print the field name and updated value
                    print(f"{field_name}: {field_value}")

                # 29. Country flag
                # Obtain the URL of the country flag by using the analysis result of Infobox.
                # # Call the function to extract the flag URL from the Infobox
                flag_url = extract_flag_url(infobox_text)

                # Print the flag URL
                print(f"\n29.Flag URL: {flag_url}")