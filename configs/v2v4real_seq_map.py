"""Explicit V2V4Real test-split scenario → sequence number mapping.

Replaces the historical map_scenarios_to_seqs() in
augment_v2v4real_with_cav_self_reports.py, which inferred the mapping from per-
scenario frame counts. That was brittle: two scenarios with the same frame count
would collide silently. This explicit map captures the V2V4Real test ordering
(seqs 0000..0008) and is the single point of truth for any tool that maps
scenarios to the AB3DMOT seqmap.

If V2V4Real ever adds/replaces test scenarios, edit THIS file (and re-checksum
v2v4real_val_label_with_cav after regenerating).
"""

from __future__ import annotations

from typing import Dict

# seq → scenario name (test__* directory under cmr_export/).
SEQ_TO_SCENARIO: Dict[int, str] = {
    0: "test__Day19__testoutput_CAV_data_2022-03-15-09-54-40_0",
    1: "test__Day19__testoutput_CAV_data_2022-03-15-10-29-43_3",
    2: "test__Day19__testoutput_CAV_data_2022-03-15-10-29-43_4",
    3: "test__Day20__testoutput_CAV_data_2022-03-17-10-50-46_0",
    4: "test__Day20__testoutput_CAV_data_2022-03-17-10-50-46_1",
    5: "test__Day20__testoutput_CAV_data_2022-03-17-11-02-23_1",
    6: "test__Day20__testoutput_CAV_data_2022-03-17-11-02-23_2",
    7: "test__Day20__testoutput_CAV_data_2022-03-17-11-51-42_0",
    8: "test__Day21__testoutput_CAV_data_2022-03-21-09-35-07_7",
}

SCENARIO_TO_SEQ: Dict[str, int] = {v: k for k, v in SEQ_TO_SCENARIO.items()}

# Per-sequence frame count (from V2V4Real's val seqmap). Used as an integrity
# check on the per-scenario cmr_export directory; mismatches must raise.
SEQ_TO_NUM_FRAMES: Dict[int, int] = {
    0: 147, 1: 114, 2: 144, 3: 198, 4: 180,
    5: 310, 6: 304, 7: 221, 8: 375,
}
