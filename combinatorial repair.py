def repair (requested_repair_type, Y_columns, all_stratified_groups, number_of_quantiles, sorted_lists, index_lookups, lambda_const):
    """
    based upon pseudocode - Michael Feldman - 2015
    [https://scholarship.tricolib.brynmawr.edu/bitstream/handle/10066/17628/2015FeldmanM.pdf?sequence=1&isAllowed=y]
    page 11

    :param requested_repair_type: geometric or combinatorial
    :param Y_columns: target columns to fix for [only one for me!]
    :param all_stratified_combinations: combos of all sensitive columns
    :param number_of_quantiles:
    :param sorted_lists:
    :param index_lookups:
    :param lambda_const: 0<lambda_const<=1
    :return:
    """

    quantile_unit = 1.0/number_of_quantiles
    for  column in Y_columns:
        group_offsets = hashmap(key: stratified group; value: offset)

        for  quantile in range(number_of_quantiles):
            median_values_at_quantile = []
            entries_at_quantile = []

            for group in all_stratified_groups:
                offset = round_to_nearest_int (group_offsets.get (group)*group.size())
                number_of_entries_to_get = round_to_nearest_int((group_offsets.get(group) + quantile_unit)*group.size()) - offset

                select entry ID and value from column, in D, where S - group, sorted by column, from offset to number_of_entries_to_get

                entries_at_quantile.append(entry IDs)
                median_values_at_quantile.append(median(values ))

            target_value = median(median_values_at_quantile)
            position_of_target = index_lookups.get(column).get(target_value)

            for  entry ID in entries_at_quantile:
                #fir each index in the quantile
                select value from column, in ID, where ID = entry ID
                #get its target value

                if requested_repair_type == "combinatorial":
                    #indexes origainl target value
                    position_of_original_value = index_lookups.get(column).get(value)
                    #distance between predicted and target value
                    distance = target_value - position_of_original_value

                    distance_to_repair = round(distance * lambda_const)
                    index_of_repair_value = position_of_original_value + distance_to_repair
                    repair_value = sorted_lists.get(col_name) [index_of_repair_value]

                else: #geometric here
                    repair_value = (1 - lambda_const)*value + lambda_const *target_value
                update D', set column = repair_value, where ID - entry ID


from itertools import product
from collections import defaultdict

from BlackBoxAuditing.repairers.AbstractRepairer import AbstractRepairer
from BlackBoxAuditing.repairers.CategoricalFeature import CategoricalFeature
from BlackBoxAuditing.repairers.calculators import get_median
from BlackBoxAuditing.repairers.SparseList import SparseList

import random
import math
from copy import deepcopy


class Repairer(AbstractRepairer):
    def repair(self, data_to_repair):
        num_cols = len(data_to_repair[0])
        col_ids = list(range(num_cols))

        # Get column type information
        col_types = ["Y"] * len(col_ids)
        for i, col in enumerate(col_ids):
            if i in self.features_to_ignore:
                col_types[i] = "I"
            elif i == self.feature_to_repair:
                col_types[i] = "X"

        col_type_dict = {col_id: col_type for col_id, col_type in zip(col_ids, col_types)}

        not_I_col_ids = [x for x in col_ids if col_type_dict[x] != "I"]

        if self.kdd:
            cols_to_repair = [x for x in col_ids if col_type_dict[x] == "Y"]
        else:
            cols_to_repair = [x for x in col_ids if col_type_dict[x] in "YX"]

        # To prevent potential perils with user-provided column names, map them to safe column names
        safe_stratify_cols = [self.feature_to_repair]

        # Extract column values for each attribute in data
        # Begin byled code will usually be created in the same directory as the .py file. initializing keys and values in dictionary
        data_dict = {col_id: [] for col_id in col_ids}

        # Populate each attribute with its column values
        for row in data_to_repair:
            for i in col_ids:
                data_dict[i].append(row[i])

        repair_types = {}
        for col_id, values in list(data_dict.items()):
            if all(isinstance(value, float) for value in values):
                repair_types[col_id] = float
            elif all(isinstance(value, int) for value in values):
                repair_types[col_id] = int
            else:
                repair_types[col_id] = str

        """
         Create unique value structures: When performing repairs, we choose median values. If repair is partial, then values will be modified to some intermediate value between the original and the median value. However, the partially repaired value will only be chosen out of values that exist in the data set.  This prevents choosing values that might not make any sense in the data's context.  To do this, for each column, we need to sort all unique values and create two data structures: a list of values, and a dict mapping values to their positions in that list. Example: There are unique_col_vals[col] = [1, 2, 5, 7, 10, 14, 20] in the column. A value 2 must be repaired to 14, but the user requests that data only be repaired by 50%. We do this by finding the value at the right index:
           index_lookup[col][2] = 1
           index_lookup[col][14] = 5
           this tells us that unique_col_vals[col][3] = 7 is 50% of the way from 2 to 14.
        """
        unique_col_vals = {}
        index_lookup = {}
        for col_id in not_I_col_ids:
            col_values = data_dict[col_id]
            # extract unique values from column and sort
            col_values = sorted(list(set(col_values)))
            unique_col_vals[col_id] = col_values
            # look up a value, get its position
            index_lookup[col_id] = {col_values[i]: i for i in range(len(col_values))}

        """
         Make a list of unique values per each stratified column.  Then make a list of combinations of stratified groups. Example: race and gender cols are stratified: [(white, female), (white, male), (black, female), (black, male)] The combinations are tuples because they can be hashed and used as dictionary keys.  From these, find the sizes of these groups.
        """
        unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
        all_stratified_groups = list(product(*unique_stratify_values))
        # look up a stratified group, and get a list of indices corresponding to that group in the data
        stratified_group_indices = defaultdict(list)

        # Find the number of unique values for each strat-group, organized per column.
        val_sets = {group: {col_id: set() for col_id in cols_to_repair}
                    for group in all_stratified_groups}
        for i, row in enumerate(data_to_repair):
            group = tuple(row[col] for col in safe_stratify_cols)
            for col_id in cols_to_repair:
                val_sets[group][col_id].add(row[col_id])

            # Also remember that this row pertains to this strat-group.
            stratified_group_indices[group].append(i)

        """
         Separate data by stratified group to perform repair on each Y column's values given that their corresponding protected attribute is a particular stratified group. We need to keep track of each Y column's values corresponding to each particular stratified group, as well as each value's index, so that when we repair the data, we can modify the correct value in the original data. Example: Supposing there is a Y column, "Score1", in which the 3rd and 5th scores, 70 and 90 respectively, belonged to black women, the data structure would look like: {("Black", "Woman"): {Score1: [(70,2),(90,4)]}}
        """
        stratified_group_data = {group: {} for group in all_stratified_groups}
        for group in all_stratified_groups:
            for col_id, col_dict in list(data_dict.items()):
                # Get the indices at which each value occurs.
                indices = {}
                for i in stratified_group_indices[group]:
                    value = col_dict[i]
                    if value not in indices:
                        indices[value] = []
                    indices[value].append(i)

                stratified_col_values = [(occurs, val) for val, occurs in list(indices.items())]
                stratified_col_values.sort(key=lambda tup: tup[1])
                stratified_group_data[group][col_id] = stratified_col_values

        mode_feature_to_repair = get_mode(data_dict[self.feature_to_repair])

        # Repair Data and retrieve the results
        for col_id in cols_to_repair:
            # which bucket value we're repairing
            group_offsets = {group: 0 for group in all_stratified_groups}
            col = data_dict[col_id]

            num_quantiles = min(len(val_sets[group][col_id]) for group in all_stratified_groups)
            quantile_unit = 1.0 / num_quantiles

            if repair_types[col_id] in {int, float}:
                for quantile in range(num_quantiles):
                    median_at_quantiles = []
                    indices_per_group = {}

                    for group in all_stratified_groups:
                        group_data_at_col = stratified_group_data[group][col_id]
                        num_vals = len(group_data_at_col)
                        offset = int(round(group_offsets[group] * num_vals))
                        number_to_get = int(round((group_offsets[group] + quantile_unit) * num_vals) - offset)
                        group_offsets[group] += quantile_unit

                        if number_to_get > 0:
                            # Get data at this quantile from this Y column such that stratified X = group
                            offset_data = group_data_at_col[offset:offset + number_to_get]
                            indices_per_group[group] = [i for val_indices, _ in offset_data for i in val_indices]
                            values = sorted([float(val) for _, val in offset_data])

                            # Find this group's median value at this quantile
                            median_at_quantiles.append(get_median(values, self.kdd))

                    # Find the median value of all groups at this quantile (chosen from each group's medians)
                    median = get_median(median_at_quantiles, self.kdd)
                    median_val_pos = index_lookup[col_id][median]

                    # Update values to repair the dataset.
                    for group in all_stratified_groups:
                        for index in indices_per_group[group]:
                            original_value = col[index]

                            current_val_pos = index_lookup[col_id][original_value]
                            distance = median_val_pos - current_val_pos  # distance between indices
                            distance_to_repair = int(round(distance * self.repair_level))
                            index_of_repair_value = current_val_pos + distance_to_repair
                            repaired_value = unique_col_vals[col_id][index_of_repair_value]

                            # Update data to repaired valued
                            data_dict[col_id][index] = repaired_value




            # Categorical Repair is done below
            elif repair_types[col_id] in {str}:
                feature = CategoricalFeature(col)
                categories = list(feature.bin_index_dict.keys())

                group_features = get_group_data(all_stratified_groups, stratified_group_data, col_id)

                categories_count = get_categories_count(categories, all_stratified_groups, group_features)

                categories_count_norm = get_categories_count_norm(categories, all_stratified_groups, categories_count,
                                                                  group_features)

                median = get_median_per_category(categories, categories_count_norm)

                # Partially fill-out the generator functions to simplify later calls.
                dist_generator = lambda group_index, category: gen_desired_dist(group_index, category, col_id, median,
                                                                                self.repair_level,
                                                                                categories_count_norm,
                                                                                self.feature_to_repair,
                                                                                mode_feature_to_repair)

                count_generator = lambda group_index, group, category: gen_desired_count(group_index, group, category,
                                                                                         median, group_features,
                                                                                         self.repair_level,
                                                                                         categories_count)

                group_features, overflow = flow_on_group_features(all_stratified_groups, group_features,
                                                                  count_generator)

                group_features, assigned_overflow, distribution = assign_overflow(all_stratified_groups, categories,
                                                                                  overflow, group_features,
                                                                                  dist_generator)

                # Return our repaired feature in the form of our original dataset
                for group in all_stratified_groups:
                    indices = stratified_group_indices[group]
                    for i, index in enumerate(indices):
                        repaired_value = group_features[group].data[i]
                        data_dict[col_id][index] = repaired_value



        # Replace stratified groups with their mode value, to remove it's information
        repaired_data = []
        for i, orig_row in enumerate(data_to_repair):
            new_row = [orig_row[j] if j not in cols_to_repair else data_dict[j][i] for j in col_ids]
            repaired_data.append(new_row)
        return repaired_data


def get_group_data(all_stratified_groups, stratified_group_data, col_id):
    group_features = {}
    for group in all_stratified_groups:
        points = [(i, val) for indices, val in stratified_group_data[group][col_id] for i in indices]
        points = sorted(points, key=lambda x: x[0])  # Sort by index
        values = [value for _, value in points]

        # send values to CategoricalFeature object, which bins the data into categories
        group_features[group] = CategoricalFeature(values)
    return group_features


# Count the observations in each category. e.g. categories_count[1] = {'A':[1,2,3], 'B':[3,1,4]}, for column 1, category 'A' has 1 observation from group 'x', 2 from 'y', ect.
def get_categories_count(categories, all_stratified_groups, group_feature):
    # Grossness for speed efficiency. Don't worry, it make me sad, too.
    count_dict = {cat: SparseList(
        data=(group_feature[group].category_count[cat] if cat in group_feature[group].category_count else 0 for group in
              all_stratified_groups)) for cat in categories}

    return count_dict


# Find the normalized count for each category, where normalized count is count divided by the number of people in that group
def get_categories_count_norm(categories, all_stratified_groups, count_dict, group_features):
    # Forgive me, Father, for I have sinned in bringing this monstrosity into being.
    norm = {cat: SparseList(
        data=(count_dict[cat][i] * (1.0 / len(group_features[group].data)) if group_features[group].data else 0.0 for
              i, group in enumerate(all_stratified_groups))) for cat in categories}
    return norm


# Find the median normalized count for each category
def get_median_per_category(categories, categories_count_norm):
    return {cat: get_median(categories_count_norm[cat], False) for cat in categories}


# Generate the desired distribution and desired "count" for a given group-category-feature combination.
def gen_desired_dist(group_index, cat, col_id, median, repair_level, norm_counts, feature_to_remove, mode):
    if feature_to_remove == col_id:
        return 1 if cat == mode else (1 - repair_level) * norm_counts[cat][group_index]
    else:
        return (1 - repair_level) * norm_counts[cat][group_index] + (repair_level * median[cat])


def gen_desired_count(group_index, group, category, median, group_features, repair_level, categories_count):
    med = median[category]
    size = len(group_features[group].data)
    # des-proportion = (1-lambda)*original-count  + (lambda)*median-count
    count = categories_count[category][group_index]
    des_count = math.floor(((1 - repair_level) * count) + (repair_level) * med * size)
    return des_count


# Run Max-flow to distribute as many observations to categories as possible. Overflow are those observations that are left over
def flow_on_group_features(all_stratified_groups, group_features, repair_generator):
    dict1 = {}
    dict2 = {}
    for i, group in enumerate(all_stratified_groups):
        feature = group_features[group]
        count_generator = lambda category: repair_generator(i, group, category)

        # Create directed graph from nodes that supply the original counts to nodes that demand
        # the desired counts, with a overflow node as total desired count is at most total original counts
        DG = feature.create_graph(count_generator)

        # Run max-flow, and record overflow count (total and per-group)
        new_feature, overflow = feature.repair(DG)
        dict2[group] = overflow

        # Update our original values with the values from max-flow, Note: still missing overflowed observations
        dict1[group] = new_feature

    return dict1, dict2


# Assign overflow observations to categories based on the group's desired distribution
def assign_overflow(all_stratified_groups, categories, overflow, group_features, repair_generator):
    feature = deepcopy(group_features)
    assigned_overflow = {}
    desired_dict_list = {}
    for group_index, group in enumerate(all_stratified_groups):
        # Calculate the category proportions.
        dist_generator = lambda cat: repair_generator(group_index, cat)
        cat_props = list(map(dist_generator, categories))

        if all(elem == 0 for elem in cat_props):  # TODO: Check that this is correct!
            cat_props = [1.0 / len(cat_props)] * len(cat_props)
        s = float(sum(cat_props))
        cat_props = [elem / s for elem in cat_props]
        desired_dict_list[group] = cat_props
        assigned_overflow[group] = {}
        for i in range(int(overflow[group])):
            distribution_list = desired_dict_list[group]
            number = random.uniform(0, 1)
            cat_index = 0
            tally = 0
            for j in range(len(distribution_list)):
                value = distribution_list[j]
                if number < (tally + value):
                    cat_index = j
                    break
                tally += value
            assigned_overflow[group][i] = categories[cat_index]
        # Actually do the assignment
        count = 0
        for i, value in enumerate(group_features[group].data):
            if value == 0:
                (feature[group].data)[i] = assigned_overflow[group][count]
                count += 1
    return feature, assigned_overflow, desired_dict_list


def get_mode(values):
    counts = {}
    for value in values:
        counts[value] = 1 if value not in counts else counts[value] + 1
    mode_tuple = max(list(counts.items()), key=lambda tup: tup[1])
    return mode_tuple[0]




if __name__ == "__main__":
    test()