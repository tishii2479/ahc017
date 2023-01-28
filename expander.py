if __name__ == "__main__":
    main_file = "src/bin/sol.rs"
    lib_file = "src/lib.rs"

    src = []
    with open(lib_file, "r") as f:
        for line in f:
            if len(line) >= 7 and line[:7] == "pub mod":
                lib_name = line.split(" ")[2][:-2]
                src.append(f"pub mod {lib_name} {{\n")
                with open(f"src/{lib_name}.rs", "r") as f:
                    is_test = False
                    for line in f:
                        if len(line) >= 7 and line[:7] == "#[test]":
                            is_test = True
                        elif is_test:
                            if len(line) >= 1 and line[:1] == "}":
                                is_test = False
                        else:
                            src.append("    " + line)
                src.append("}\n")
            else:
                src.append(line)

    src.append("\n")

    with open(main_file, "r") as f:
        for line in f:
            src.append(line.replace("ahc016", "crate"))

    for line in src:
        print(line, end="")
