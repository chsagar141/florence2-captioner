import os

# Folder containing the .txt files
folder_path = "Rename"
prepend_text = "Candle_Light , "

# Ensure the folder exists
if not os.path.isdir(folder_path):
    print(f"❌ Folder '{folder_path}' not found.")
else:
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Prepend only if not already added
            if not content.startswith(prepend_text):
                new_content = prepend_text + content
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(new_content)
                print(f"✅ Updated: {filename}")
            else:
                print(f"ℹ️ Already updated: {filename}")
