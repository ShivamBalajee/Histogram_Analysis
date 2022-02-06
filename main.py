


# !/usr/local/bin/python3
#
# Authors: Marcus Skinner (marcskin) Parth Verma (paverma) Shivam Balajee (shbala)
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from PIL import Image
import numpy as np
from scipy.ndimage import filters
import sys
import imageio


# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = np.array(input_image.convert('L'))
    filtered_y = np.zeros(grayscale.shape)
    filters.sobel(grayscale, 0, filtered_y)
    return np.sqrt(filtered_y ** 2)


def transmission_probability(prev_row, current_row, s=10):
    return np.exp(-abs(prev_row - current_row) / s)


# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range(int(max(y - int(thickness / 2), 0)), int(min(y + int(thickness / 2), image.size[1] - 1))):
            image.putpixel((x, t), color)
    return image


def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [(pt[0] + dx, pt[1] + dy) for dx in range(-3, 4) for dy in range(-2, 3) if
                   dx == 0 or dy == 0 or abs(dx) == abs(dy)]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)


# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception(
            "Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [int(i) for i in sys.argv[2:4]]
    gt_icerock = [int(i) for i in sys.argv[4:6]]

    # load in image
    input_image = Image.open(input_filename).convert('RGB')
    image_array = np.array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', np.uint8(255 * edge_strength / (np.amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    max_edge_strength = np.max(edge_strength)
    edge_strength = edge_strength / max_edge_strength

    airice_simple = []

    for i in range(edge_strength.shape[1]):
        column = edge_strength[:, i]
        s = 11
        ans = []
        for j in range(column.shape[0] - s):
            ans.append(np.mean(column[j:j + s]))
        airice_simple.append(np.argmax(ans) + s // 2 + 1)

    # airice_simple = [image_array.shape[0] * 0.25] * image_array.shape[1]
    airice_hmm = []

    viterbi_table = np.zeros(edge_strength.shape) - 999
    which_table = np.zeros(edge_strength.shape)
    s = 7
    for i in range(s // 2, viterbi_table.shape[0] - s // 2):
        viterbi_table[i][0] = np.log(np.mean(edge_strength[i - s // 2:i + s // 2 + 1, 0]))

    for column_index in range(1, edge_strength.shape[1]):
        for row_index in range(s // 2, edge_strength.shape[0]):
            which_table[row_index][column_index], viterbi_table[row_index][column_index] = max(
                [(row_index_prev, prob + np.log(transmission_probability(row_index, row_index_prev))) for
                 row_index_prev, prob
                 in enumerate(viterbi_table[:, column_index - 1])], key=lambda x: x[1])
            viterbi_table[row_index][column_index] += np.log(np.mean(
                edge_strength[row_index - s // 2:row_index + s // 2 + 1, column_index]))

    airice_hmm = [0] * edge_strength.shape[1]
    airice_hmm[-1] = np.argmax(viterbi_table[:, -1])
    for i in range(edge_strength.shape[1] - 2, -1, -1):
        airice_hmm[i] = int(which_table[airice_hmm[i + 1], i + 1])
    # airice_hmm = [image_array.shape[0] * 0.5] * image_array.shape[1]

    # feed back
    air_ice_row, air_ice_col = gt_airice
    viterbi_table[:, air_ice_col] = -999
    viterbi_table[air_ice_row, air_ice_col] = 0
    which_table[air_ice_row, air_ice_col] = max(
        [(row_index_prev, prob + np.log(transmission_probability(air_ice_row, row_index_prev))) for row_index_prev, prob
         in enumerate(viterbi_table[:, air_ice_col - 1])], key=lambda x: x[1])[0]

    for column_index in range(air_ice_col + 1, edge_strength.shape[1]):
        for row_index in range(s // 2, edge_strength.shape[0]):
            which_table[row_index][column_index], viterbi_table[row_index][column_index] = max(
                [(row_index_prev, prob + np.log(transmission_probability(row_index, row_index_prev))) for
                 row_index_prev, prob
                 in enumerate(viterbi_table[:, column_index - 1])], key=lambda x: x[1])
            viterbi_table[row_index][column_index] += np.log(np.mean(
                edge_strength[row_index - s // 2:row_index + s // 2 + 1, column_index]))
    airice_feedback = [0] * edge_strength.shape[1]
    airice_feedback[-1] = np.argmax(viterbi_table[:, -1])
    for i in range(edge_strength.shape[1] - 2, -1, -1):
        airice_feedback[i] = int(which_table[airice_feedback[i + 1], i + 1])

    # airice_feedback = [image_array.shape[0] * 0.75] * image_array.shape[1]

    icerock_simple = []

    for i in range(edge_strength.shape[1]):
        column = edge_strength[:, i]
        s = 5
        ans = []
        for j in range(airice_simple[i] + s // 2 + 10, column.shape[0] - s):
            ans.append(np.mean(column[j:j + s]))
        icerock_simple.append(np.argmax(ans) + s // 2 + 1 + airice_simple[i] + 10)
    # icerock_simple = [image_array.shape[0] * 0.25] * image_array.shape[1]
    # icerock_hmm = [image_array.shape[0] * 0.5] * image_array.shape[1]

    icerock_hmm = []

    viterbi_table = np.zeros(edge_strength.shape) - 999
    which_table = np.zeros(edge_strength.shape)
    s = 1
    for i in range(s // 2, viterbi_table.shape[0] - s // 2):
        viterbi_table[i][0] = np.log(np.mean(edge_strength[i - s // 2:i + s // 2 + 1, 0]))

    for column_index in range(1, edge_strength.shape[1]):
        for row_index in range(s // 2, edge_strength.shape[0]):
            if row_index <= airice_hmm[column_index] + 10:
                continue
            which_table[row_index][column_index], viterbi_table[row_index][column_index] = max(
                [(row_index_prev, prob + np.log(transmission_probability(row_index, row_index_prev, 4))) for
                 row_index_prev, prob
                 in enumerate(viterbi_table[:, column_index - 1])], key=lambda x: x[1])
            viterbi_table[row_index][column_index] += np.log(np.mean(
                edge_strength[row_index - s // 2:row_index + s // 2 + 1, column_index]))

    icerock_hmm = [0] * edge_strength.shape[1]
    icerock_hmm[-1] = np.argmax(viterbi_table[:, -1])
    for i in range(edge_strength.shape[1] - 2, -1, -1):
        icerock_hmm[i] = int(which_table[icerock_hmm[i + 1], i + 1])

    ##
    # s=3
    ice_rock_col, ice_rock_row = gt_icerock
    viterbi_table[:, ice_rock_col] = -999
    viterbi_table[ice_rock_row, ice_rock_col] = 0
    which_table[ice_rock_row, ice_rock_col] = max(
        [(row_index_prev, prob + np.log(transmission_probability(ice_rock_row, row_index_prev))) for
         row_index_prev, prob
         in enumerate(viterbi_table[:, ice_rock_col - 1])], key=lambda x: x[1])[0]

    for column_index in range(ice_rock_col + 1, edge_strength.shape[1]):
        for row_index in range(s // 2, edge_strength.shape[0]):
            if row_index <= airice_hmm[column_index] + 10:
                continue
            which_table[row_index][column_index], viterbi_table[row_index][column_index] = max(
                [(row_index_prev, prob + np.log(transmission_probability(row_index, row_index_prev))) for
                 row_index_prev, prob
                 in enumerate(viterbi_table[:, column_index - 1])], key=lambda x: x[1])
            viterbi_table[row_index][column_index] += np.log(np.mean(
                edge_strength[row_index - s // 2:row_index + s // 2 + 1, column_index]))
    icerock_feedback = [0] * edge_strength.shape[1]
    icerock_feedback[-1] = np.argmax(viterbi_table[:, -1])
    for i in range(edge_strength.shape[1] - 2, -1, -1):
        icerock_feedback[i] = int(which_table[icerock_feedback[i + 1], i + 1])

    # icerock_feedback = [image_array.shape[0] * 0.75] * image_array.shape[1]

    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")
