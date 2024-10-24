def generate_square_coordinates2(canvas_size, square_size, pattern):
    coordinates = []
    y_offset = (canvas_size[1] - (len(pattern) * square_size)) // (len(pattern) + 1)
    current_y = y_offset

    for row in pattern:
        x_offset = (canvas_size[0] - (row * square_size)) // (row + 1)
        current_x = x_offset

        for _ in range(row):
            coordinates.append((current_x, current_y))
            current_x += square_size + x_offset

        current_y += square_size + y_offset

    return coordinates


def generate_square_coordinates(canvas_size, square_size, pattern):
    coordinates = []
    y_offset = (canvas_size[1] - (len(pattern) * square_size)) // (len(pattern) + 1)
    current_y = y_offset

    for row in pattern:
        if row == 0:
            # Skip this row if there are no squares to draw
            current_y += square_size + y_offset
            continue

        x_offset = (canvas_size[0] - (row * square_size)) // (row + 1)
        current_x = x_offset

        for _ in range(row):
            coordinates.append((current_x, current_y))
            current_x += square_size + x_offset

        current_y += square_size + y_offset

    return coordinates


def get_roi_from_file(path: str):
    """
    Helper function to read bias from a file
    """
    roi = {}
    try:
        roi_file = open(path, "r")
    except IOError:
        print("Cannot open roi file: " + path)
    else:
        for line in roi_file:
            # Skip lines starting with '%': comments
            if line.startswith("%"):
                continue

            split_line = line.split(" ")
            if len(split_line) == 4:
                roi["x"] = int(split_line[0])
                roi["y"] = int(split_line[1])
                roi["width"] = int(split_line[2])
                roi["height"] = int(split_line[3])

    return roi


def get_biases_from_file(path: str):
    """
    Helper function to read bias from a file
    """
    biases = {}
    try:
        biases_file = open(path, "r")
    except IOError:
        print("Cannot open bias file: " + path)
    else:
        for line in biases_file:
            # Skip lines starting with '%': comments
            if line.startswith("%"):
                continue

            # element 0 : value
            # element 1 : name
            split = line.split("%")
            biases[split[1].strip()] = int(split[0])
    return biases



