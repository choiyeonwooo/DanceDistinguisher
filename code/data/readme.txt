The data format looks like the example below

example_data = (num_frames, [x and y coordianates per frame (total 66, values are between 0 and 1)])
all example data are stored in a single file which is a type of pickle file.

data = {
    "fname" : "example.mp4",
    "coordinates": (num_frames, x-y coordinates),
    "label" : 0 to 9, integer value
}