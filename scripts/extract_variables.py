import glob
import json

from tqdm import tqdm

dataset_path = "data/train/"

def extract_variables(program):
    locations = set()
    tokens = program["source_tokens"]
    for loc in program["repair_candidates"]:
        if type(loc) == int:
            locations.add(loc)

    locations.add(program["error_location"])

    variables = dict()
    for loc in locations:
        variable = tokens[loc]
        variables[variable] = variables.get(variable, 0) + 1

    return variables


def main():
    filenames = sorted(glob.glob(dataset_path + "*VARIABLE_MISUSE__SStuB*"))
    variables_map = dict()

    for filename in tqdm(filenames):
        programs = []
        with open(filename, "r") as file:
            for line in file:
                programs.append(json.loads(line))

        for program in programs:
            variables = extract_variables(program)
            for variable in variables:
                variables_map[variable] = variables_map.get(variable, 0) + 1

    with open("variables_map.json", "w") as file:
        json.dump(variables_map, file, indent=4)


if __name__ == "__main__":
    main()
