# import file
import pandas as pd
from colormath.color_objects import sRGBColor, LabColor, HSLColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import itertools
import random
import os
import numpy
import matplotlib.pyplot as plt
import colorsys
import seaborn as sns

def patch_asscalar(a):
    return a.item()

random.seed(10)

setattr(numpy, "asscalar", patch_asscalar)

# Read the CSV file
df = pd.read_csv(os.path.join(os.getcwd(),'mb5_color_palette.csv'))

#convert hsl values (from specified range in color palette csv)
def convert_hsl(hsl):
    return [hsl[0]/255*360,hsl[1]/255,hsl[2]/255]

# Convert HSL colors to Lab colors for perceptual similarity calculations
def hsl_to_lab(hsl):
    hsl_color = HSLColor(hsl[0], hsl[1], hsl[2])
    rgb_color = convert_color(hsl_color, sRGBColor)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color

# Calculate perceptual similarity (Delta E) between two colors
def calculate_delta_e(color1, color2):
    return delta_e_cie2000(color1, color2)

# Get the list of colors from the dataframe
unconverted_colors = df[['h', 's', 'l']].values
print(unconverted_colors)
colors = [convert_hsl(color) for color in unconverted_colors]
print(colors)
lab_colors = [hsl_to_lab(color) for color in colors]

# Calculate the perceptual similarity matrix
similarity_matrix = [[calculate_delta_e(c1, c2) for c2 in lab_colors] for c1 in lab_colors]

# Generate all possible combinations of 5 colors
all_combinations = list(itertools.combinations(range(len(colors)), 5))

#function to determine the smallest perceptual similarity within a combination
def smallest_similarity_in_combination(combination, similarity_matrix):
    min_similarity = float('inf')
    for i, j in itertools.combinations(combination, 2):
        similarity = similarity_matrix[i][j]
        if similarity < min_similarity:
            min_similarity = similarity
    return min_similarity

#compute minimum similarities
min_similarities = [smallest_similarity_in_combination(comb, similarity_matrix) for comb in all_combinations]

#inspect thos minimum similarities
# plt.hist(min_similarities, bins=30, edgecolor='black')
# plt.title('Distribution of Minimum Similarity Values')
# plt.xlabel('Minimum Similarity Value')
# plt.ylabel('Frequency')
# plt.show()

#let's enforce a difference of at least 20 dE
# Filter combinations to ensure the smallest similarity value exceeds 20
filtered_combinations = [comb for comb in all_combinations if smallest_similarity_in_combination(comb, similarity_matrix) > 20]

# Function to calculate the distinctiveness of a combination
def combination_distinctiveness(combination):
    return sum(similarity_matrix[i][j] for i, j in itertools.combinations(combination, 2))

# Select the 12 most distinctive combinations
#distinctive_combinations = sorted(filtered_combinations, key=combination_distinctiveness, reverse=True)[:12]
# Select the most distinctive combinations from the filtered list
distinctive_combinations = sorted(filtered_combinations, key=combination_distinctiveness, reverse=True)


# Ensure each color is the "base" color once and each combination contains 5 different colors
import random

# Ensure each color is the "base" color once and each combination contains 5 different colors
final_combinations = []
used_colors = [0] * len(colors)
max_usage_difference = 2

for base_color in range(12):  # Assuming there are 12 distinct colors
    valid_combinations = [comb for comb in distinctive_combinations if base_color in comb]
    print(len(valid_combinations))
    while valid_combinations:
        selected_combination = random.choice(valid_combinations)
        final_combination = [base_color] + [color for color in selected_combination if color != base_color]
        
        # Check if adding this combination keeps the color usage balanced
        temp_used_colors = used_colors[:]
        for color in final_combination:
            temp_used_colors[color] += 1
        
        if max(temp_used_colors) - min(temp_used_colors) <= max_usage_difference:
            final_combinations.append(final_combination)
            used_colors = temp_used_colors
            break
        else:
            valid_combinations.remove(selected_combination)

print(len(final_combinations))

# If we don't have 12 combinations, continue to find more
if len(final_combinations) < 12:
    for combination in distinctive_combinations:
        if len(final_combinations) >= 12:
            break
        for base_color in range(12):
            if base_color not in [comb[0] for comb in final_combinations] and base_color in combination:
                final_combination = [base_color] + [color for color in combination if color != base_color]
                
                # Check if adding this combination keeps the color usage balanced
                temp_used_colors = used_colors[:]
                for color in final_combination:
                    temp_used_colors[color] += 1
                
                if max(temp_used_colors) - min(temp_used_colors) <= max_usage_difference+1:
                    final_combinations.append(final_combination)
                    used_colors = temp_used_colors
                else:
                    # If we can't keep the usage balanced, still add the combination to reach 12
                    final_combinations.append(final_combination)
                    used_colors = temp_used_colors
                break

# Ensure we have exactly 12 combinations
while len(final_combinations) < 12:
    for combination in distinctive_combinations:
        if len(final_combinations) >= 12:
            break
        for base_color in range(12):
            if base_color not in [comb[0] for comb in final_combinations] and base_color in combination:
                final_combination = [base_color] + [color for color in combination if color != base_color]
                final_combinations.append(final_combination)
                for color in final_combination:
                    used_colors[color] += 1
                break

# Print the final combinations
for i, combination in enumerate(final_combinations):
    print(f"Combination {i+1}: {[colors[idx] for idx in combination]}")

print("\nFinal color usage breakdown:")
for i, usage in enumerate(used_colors):
    print(f"Color {i}: {usage} times")

def hsl_to_rgb(h, s, l):
    h = h /360
    return colorsys.hls_to_rgb(h, l, s)

# Create a figure and axes
# fig, ax = plt.subplots()

# current_combo=[colors[idx] for idx in final_combinations[0]]

# # Iterate over HSL values and plot color patches
# for color in current_combo:
#     rgb = hsl_to_rgb(color[0], color[1], color[2])
#     print(rgb)
#     ax.add_patch(plt.Rectangle((color[0]/360, color[1]), 0.1, -0.2, color=rgb))

# plt.ylim(0,1.2)

# # Set axis labels
# ax.set_xlabel('Hue')
# ax.set_ylabel('Saturation')
# ax.set_title('Color Combinations')

# # Show the plot
# plt.show()

palette_counter = 1

for current_combo in final_combinations:
    cur_color_list =[colors[idx] for idx in current_combo]
    my_palette = [hsl_to_rgb(color[0], color[1], color[2]) for color in cur_color_list]
    print(my_palette)
    # Display the palette
    sns.palplot(my_palette)
    # plt.show()
    plt.savefig("mb5_palette_"+str(palette_counter)+".png")
    palette_counter += 1