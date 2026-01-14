#!/usr/bin/env python


# Import modules
import columnar

# Name of object to display attributes for
module_name="logging"
# Import module by name in str var
logging = __import__(module_name)
# module = importlib.import_module(module_name, package=None)

# Define global constants
debug = False
# debug = True


# Initialize local variables
no_borders = True
headers = ['Attribute','Attribute Value']
content = []
log_dict = logging.__dict__
keys = log_dict.keys()

# parse attributes from target object
for key in keys:
    # Ignore private attributes
    if not key.startswith("_"):
        content_row = [key, log_dict.get(key)]
        print(f"Content Row: {content_row}") if debug else None
        content.append(content_row,)
        print(f"Content: {content}") if debug else None
    else:
        print(f"Skipping private attribute: {key}") if debug else None

print(f"Final Content: {content}") if debug else None

# generate the columnar table
column_table = columnar.columnar(data=content, headers=headers, no_borders=no_borders)
# Display the columnar table
print(f"{column_table}")

