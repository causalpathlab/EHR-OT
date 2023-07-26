#!/usr/bin/env python
# coding: utf-8

# In[3]:


from ast import literal_eval
import pandas as pd


def construct_freq_dict_group(df, dividing_feature, group_1, group_2):
    """ 
    Construct frequency dictionary dictionary for group 1 and group 2,
    considering the duplicate codes for one admission as one code
    """

    group_1_df = df[df[dividing_feature] == group_1]
    group_2_df = df[df[dividing_feature] == group_2]

    def construct_freq_dict(group_df):
        group_dict = {}
        for _, row in group_df.iterrows():
            code_set = list(set(row['ICD codes']))
            for code in code_set:
                if code not in group_dict:
                    group_dict[code] = 1
                else:
                    group_dict[code] += 1
        return group_dict
    
    return construct_freq_dict(group_1_df), construct_freq_dict(group_2_df)



# In[16]:


# Filter out ICD codes with two few counts

def select_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count):
    """ 
    Select codes for group 1 and group 2 using their minimum count
    """

    all_codes = list(group_1_dict.keys())
    group_2_codes = list(group_2_dict.keys())
    all_codes.extend(group_2_codes)
    all_codes = list(set(all_codes))

    def filter_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count):
        """ 
        Codes filtered for group 1 and group 2 using their minimum count
        """
        def filtered_code_group(group_dict, min_count):
            """ 
            Codes filtered out for a group since the number is too few
            """
            filtered_codes = []
            for code, value in group_dict.items():
                if value < min_count:
                    filtered_codes.append(code)
            return filtered_codes
        filtered_group_1_codes = filtered_code_group(group_1_dict, group_1_min_count)
        filtered_group_2_codes = filtered_code_group(group_2_dict, group_2_min_count)
        filtered_codes = filtered_group_1_codes
        filtered_codes.extend(filtered_group_2_codes)
        return list(set(filtered_codes))

    filtered_codes = filter_codes(group_1_dict, group_2_dict, group_1_min_count, group_2_min_count)
    selected_codes = [x for x in all_codes if x not in filtered_codes]
    return selected_codes





