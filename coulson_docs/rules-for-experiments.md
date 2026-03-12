Every experiment (every single benchmark client launch) need to have a unique experiment id, for diagmoe, it should be amoe-<idnumber>, for sglang, it should be sgl-<idnumber>

All the log files, metric logging, stats, should be aggregated at the head node's <PROJECT_ROOT>/experiments dir, under a subdir named by the experiment id.
Those individual experiments dir should be ignored by git.
The plots and plotting scripts associated with it should all be inside <PROJECT_ROOT>/experiments as well, these should not be ignored by git.