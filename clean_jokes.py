import re

def clean_jokes(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            jokes = file.readlines()

        if not jokes:
            print("Warning: The input file is empty or not read correctly.")
        
        cleaned_jokes = []
        for joke in jokes:
            
            joke = joke.strip()
            
            joke = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\']', '', joke)
            if joke:  
                cleaned_jokes.append(joke)
        
        if not cleaned_jokes:
            print("Warning: No jokes were cleaned.")
        
        with open(output_path, 'w', encoding='utf-8') as file:
            for joke in cleaned_jokes:
                file.write(f"{joke}\n")

        print(f"Cleaned jokes dataset has been saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

input_file_path = "jokes.txt"

output_file_path = "cleaned_jokes.txt"


clean_jokes(input_file_path, output_file_path)
