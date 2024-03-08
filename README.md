# Transport-based transfer learning on Electronic Health Records throught optimal transport (OTTEHR): Application to detection of treatment disparities #

Run run_OTTEHR.py to run OTTEHR with **group_name** and **groups** updated to the appropriate values. 

Run mimic_exp/admission/analyze_bound.ipynb to generate analysis of target error and individual terms in the derived upper bound.

Run mimic_exp/admission/analyze_accuracy.ipynb to generate benchmarking results of OTTEHR against existing transfer learning methods. 

Run mimic_exp/admission/analyze_duration_diff.ipynb to generate predicted duration vs observed duration and the treatment disparity analysis based on subgroups.