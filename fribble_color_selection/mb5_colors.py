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

setattr(numpy, "asscalar", patch_asscalar)

#convert hsl values (from specified range in color palette csv)
def convert_hsl(hsl):
    return [hsl[0]/255*360,hsl[1]/255,hsl[2]/255]

#convert hsl to rgb
def hsl_to_rgb(h, s, l):
    h = h /360
    return colorsys.hls_to_rgb(h, l, s)

#flatten lists
def flatten_list_comprehension(nested_list):
    return [item for sublist in nested_list for item in sublist]

# Convert HSL colors to Lab colors for perceptual similarity calculations
def hsl_to_lab(hsl):
    hsl_color = HSLColor(hsl[0], hsl[1], hsl[2])
    rgb_color = convert_color(hsl_color, sRGBColor)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color

# Calculate perceptual similarity (Delta E) between two colors
def calculate_delta_e(color1, color2):
    return delta_e_cie2000(color1, color2)

#function to generate color combinations
def generate_combinations(lst):
    n = len(lst)
    result = []
    #alternating index
    alternator = 0
    
    for i in range(n): 
        alternator = (alternator + 1) % 2
        combination = []
        index = i
        while len(combination) < 5:
            alternator = (alternator + 1) % 2
            print(alternator)
            combination.append(lst[index])
            if alternator == 1:
                index = (index + 2) % n  # Skip adjacent items, wrap around if needed
            elif alternator == 0:
                index = (index + 3) % n  # Skip 2 adjacent items, wrap around if needed
        result.append(combination)
    
    return result

#find the minimum values in nested lists
def find_min_values(nested_lists):
    result = []
    
    for lst in nested_lists:
        # Filter out first element
        filtered_lst = lst[1:]

        if not filtered_lst:  # Handle empty lists
            result.append({"min_value": None, "index": None})
        else:
            min_value = min(filtered_lst)
            min_index = lst.index(min_value)
            result.append({"min_value": min_value, "index": min_index})
    
    return result

#function to put the max color item in the second position of the palette list
def find_max_values_and_put_second_combos(nested_lists):
    result = []
    
    for lst in nested_lists:

        if not lst:  # Handle empty lists
            result.append(lst)  # Keep original order
            #result.append({"max_value": None, "index": None})
            continue
        else:
            max_value = max(lst)
            max_index = lst.index(max_value)
            #result.append({"max_value": max_value, "index": max_index})
    
            # Reorder list so the minimum value is in the second position
            new_index_list = [0]  # Keep the first element unchanged
            new_index_list.append(max_index)  # Move max value to second position
            remaining_elements = [i for i in range(1, len(lst)) if i != max_index]  # Exclude the max value
            new_index_list.extend(remaining_elements)  # Append the rest of the elements

            result.append(new_index_list)

    return result

def reorder_list(lst, indices):
    """
    Reorders a list based on a given list of indices.
    
    Args:
        lst (list): The original list.
        indices (list): A list of new positions for the elements.
    
    Returns:
        list: The reordered list.
    """
    if len(lst) != len(indices):
        raise ValueError("The list and indices must be the same length")
    
    return [lst[i] for i in indices]

#### Begin script

random.seed(10)

# Read the CSV file
df = pd.read_csv(os.path.join(os.getcwd(),'data','mb5_color_palette.csv'))

# Get the list of colors from the dataframe
color_names = df['color_name'].tolist()
h_values = df['h'].tolist()
s_values = df['s'].tolist()
l_values = df['l'].tolist()
unconverted_colors = df[['h', 's', 'l']].values
print(unconverted_colors)
colors = [convert_hsl(color) for color in unconverted_colors]
print(colors)
lab_colors = [hsl_to_lab(color) for color in colors]
color_index = [i  for i in range(len(colors))]

#generate a list of color combinations (indices)
combinations_list = generate_combinations(color_index)
print(combinations_list)

color_combinations = [[colors[idx] for idx in current_combo] for current_combo in combinations_list]
lab_color_combinations = [[lab_colors[idx] for idx in current_combo] for current_combo in combinations_list]
print(color_combinations)
print(len(color_combinations))
print(lab_color_combinations)

# Calculate the perceptual similarity matrix
similarity_matrix = [flatten_list_comprehension([[calculate_delta_e(c1, c2) for c2 in lab_color_list] for c1 in lab_color_list]) for lab_color_list in lab_color_combinations]
print(similarity_matrix)

base_difference_matrix = [[calculate_delta_e(lab_color_list[0], c2) for c2 in lab_color_list] for lab_color_list in lab_color_combinations]
print(base_difference_matrix)

# list max similarities
new_indices_list = find_max_values_and_put_second_combos(base_difference_matrix)

new_combinations_list = [reorder_list(combinations_list[i],new_indices_list[i]) for i in range(len(combinations_list))]

# Print results
for i, info in enumerate(new_combinations_list):
    print(f"List {i+1}: {info}")


palette_counter = 1
all_colors_data=[]
for current_combo in new_combinations_list:
    cur_color_name_list = [color_names[idx] for idx in current_combo]
    cur_h_values_list=[h_values[idx] for idx in current_combo]
    cur_s_values_list=[s_values[idx] for idx in current_combo]
    cur_l_values_list=[l_values[idx] for idx in current_combo]
    cur_color_list =[colors[idx] for idx in current_combo]
    my_palette = [hsl_to_rgb(color[0], color[1], color[2]) for color in cur_color_list]
    #print(my_palette)
    
    # Append data for each color in the current combination
    appendage_counter=1
    for i in range(len(current_combo)):
        if i == 0:
            cur_type = "base_color"
        else:
            cur_type = "appendage_"+str(appendage_counter)
            appendage_counter+=1
        
        all_colors_data.append({
            "combination": palette_counter,
            "type": cur_type,
            "color_name": cur_color_name_list[i],
            "h_value": cur_h_values_list[i],
            "s_value": cur_s_values_list[i],
            "l_value": cur_l_values_list[i],
            "hsl_color": cur_color_list[i],
            "rgb_color": my_palette[i]
        })

    # Display the palette
    sns.palplot(my_palette)
    # plt.show()
    plt.savefig(os.path.join(os.getcwd(),"palette_combos","mb5_palette_"+str(palette_counter)+".png"))
    palette_counter += 1


#write data
df_colors = pd.DataFrame(all_colors_data)
# Save DataFrame to CSV
df_colors.to_csv(os.path.join(os.getcwd(),'data',"mb5_color_combinations.csv"), index=False) 



