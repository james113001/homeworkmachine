import os
import json
import pandas as pd
import re

def load_problems_from_directory(directory):
    problems = []
    for subdir, _, files in os.walk(directory):
        problem_type = os.path.basename(subdir)
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r') as f:
                        problem = json.load(f)
                        problem['type'] = problem_type
                        answer = extract_answer(problem.get('solution'))
                        if is_valid_answer(answer):
                            problem['answer'] = answer
                            problems.append(problem)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    return problems

def extract_answer(solution):
    # This regex matches the LaTeX boxed answer, allowing for nested braces
    match = re.search(r'\\boxed{((?:[^{}]|{(?:[^{}]|{[^{}]*})*})*)}', solution)
    if match:
        return match.group(1)
    return None

def is_valid_answer(answer):
    # Check if the answer is an integer or an integer with a LaTeX unit
    # This regex checks for an optional sign, followed by one or more digits, optionally followed by a LaTeX unit
    if re.match(r'^-?\d+(\s*\\\w+)?$', answer):
        return True
    return False

def compile_dataset(directory, output_csv):
    problems = load_problems_from_directory(directory)

    data = {
        'Question': [],
        'Level': [],
        'Type': [],
        'Solution': [],
        'Answer': []
    }

    for problem in problems:
        data['Question'].append(problem.get('problem'))
        data['Level'].append(problem.get('level'))
        data['Type'].append(problem.get('type'))
        data['Solution'].append(problem.get('solution'))
        data['Answer'].append(problem.get('answer'))

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(problems)} problems to {output_csv}")

def main():
    # Compile train dataset
    train_directory = 'MATH/train'  # Update this path if necessary
    train_output_csv = 'traincompiled.csv'
    compile_dataset(train_directory, train_output_csv)

    # Compile test dataset
    test_directory = 'MATH/test'  # Update this path if necessary
    test_output_csv = 'testcompiled.csv'
    compile_dataset(test_directory, test_output_csv)

if __name__ == "__main__":
    main()
