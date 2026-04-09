# sign-language-bridge
Real-time American Sign Language (ASL) to English translation using a Transformer encoder-decoder architecture.



The details of the architicure.
weights of the MLP and encoder are initialized from 0.
data augmentation.




=======================================================
  OpenASL -- Final Processed Dataset
=======================================================

  TRAIN: 39,331 clips
    Duration  : 1.3s - 8.0s  (mean 4.4s)
    Frames    : 26 - 160  (mean 89)
    Total hrs : 48.5h
    Words/clip: 9.9 avg

  VAL: 461 clips
    Duration  : 1.3s - 8.0s  (mean 4.3s)
    Frames    : 27 - 160  (mean 86)
    Total hrs : 0.6h
    Words/clip: 10.5 avg

  TEST: 428 clips
    Duration  : 1.3s - 8.0s  (mean 4.7s)
    Frames    : 27 - 160  (mean 94)
    Total hrs : 0.6h
    Words/clip: 10.8 avg



============================================================
DATASET COMBINATION SCRIPT
============================================================

This script combines How2Sign and OpenASL datasets
into unified train/val/test splits with a common schema.


████████████████████████████████████████████████████████████
TRAIN SPLIT
████████████████████████████████████████████████████████████

Reading How2Sign train...
Original columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'row_duration_sec', 'duration_sec', 'word_count']

============================================================
How2Sign train (ORIGINAL)
============================================================
Number of examples: 19,935
Columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'row_duration_sec', 'duration_sec', 'word_count']

Duration statistics:
  Total duration: 86407.39 seconds (24.00 hours)
  Mean duration: 4.33 seconds
  Median duration: 4.18 seconds
  Min duration: 1.30 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 261,165
  Mean words per example: 13.10
  Median words: 12
  Min words: 1
  Max words: 54

Reading OpenASL train...
Original columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

============================================================
OpenASL train (ORIGINAL)
============================================================
Number of examples: 39,331
Columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

Duration statistics:
  Total duration: 174775.52 seconds (48.55 hours)
  Mean duration: 4.44 seconds
  Median duration: 4.34 seconds
  Min duration: 1.30 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 389,388
  Mean words per example: 9.90
  Median words: 9
  Min words: 1
  Max words: 46

Text statistics:
  Mean text length: 53.76 characters
  Median text length: 50 characters

============================================================
COMBINING TRAIN SPLIT
============================================================

How2Sign train: 19,935 examples
OpenASL train: 39,331 examples
Combined total: 59,266 examples

============================================================
COMBINED TRAIN (AFTER MERGING)
============================================================
Number of examples: 59,266
Columns: ['vid', 'text', 'start', 'end', 'duration_sec', 'word_count']

Duration statistics:
  Total duration: 261182.91 seconds (72.55 hours)
  Mean duration: 4.41 seconds
  Median duration: 4.28 seconds
  Min duration: 1.30 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 650,553
  Mean words per example: 10.98
  Median words: 10
  Min words: 1
  Max words: 54

Text statistics:
  Mean text length: 58.08 characters
  Median text length: 54 characters

✓ Saved to: C:\My Projects\sign-language-bridge\data\final_full_train_dataset.tsv


████████████████████████████████████████████████████████████
VALIDATION SPLIT
████████████████████████████████████████████████████████████

Reading How2Sign validation...
Original columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'duration_sec']
  Calculated word_count from text (was missing)

============================================================
How2Sign validation (ORIGINAL)
============================================================
Number of examples: 1,051
Columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'duration_sec']

Duration statistics:
  Total duration: 4593.68 seconds (1.28 hours)
  Mean duration: 4.37 seconds
  Median duration: 4.27 seconds
  Min duration: 1.31 seconds
  Max duration: 8.00 seconds

Reading OpenASL validation...
Original columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

============================================================
OpenASL validation (ORIGINAL)
============================================================
Number of examples: 461
Columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

Duration statistics:
  Total duration: 1990.16 seconds (0.55 hours)
  Mean duration: 4.32 seconds
  Median duration: 4.27 seconds
  Min duration: 1.33 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 4,863
  Mean words per example: 10.55
  Median words: 10
  Min words: 1
  Max words: 30

Text statistics:
  Mean text length: 57.82 characters
  Median text length: 55 characters

============================================================
COMBINING VALIDATION SPLIT
============================================================

How2Sign validation: 1,051 examples
OpenASL validation: 461 examples
Combined total: 1,512 examples

============================================================
COMBINED VALIDATION (AFTER MERGING)
============================================================
Number of examples: 1,512
Columns: ['vid', 'text', 'start', 'end', 'duration_sec', 'word_count']

Duration statistics:
  Total duration: 6583.84 seconds (1.83 hours)
  Mean duration: 4.35 seconds
  Median duration: 4.27 seconds
  Min duration: 1.31 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 18,194
  Mean words per example: 12.03
  Median words: 11
  Min words: 1
  Max words: 36

Text statistics:
  Mean text length: 62.47 characters
  Median text length: 59 characters

✓ Saved to: C:\My Projects\sign-language-bridge\data\final_full_val_dataset.tsv


████████████████████████████████████████████████████████████
TEST SPLIT
████████████████████████████████████████████████████████████

Reading How2Sign test...
Original columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'duration_sec']
  Calculated word_count from text (was missing)

============================================================
How2Sign test (ORIGINAL)
============================================================
Number of examples: 1,464
Columns: ['VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED', 'SENTENCE', 'duration_sec']

Duration statistics:
  Total duration: 6288.22 seconds (1.75 hours)
  Mean duration: 4.30 seconds
  Median duration: 4.16 seconds
  Min duration: 1.31 seconds
  Max duration: 8.00 seconds

Reading OpenASL test...
Original columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

============================================================
OpenASL test (ORIGINAL)
============================================================
Number of examples: 428
Columns: ['vid', 'yid', 'start', 'end', 'text', 'duration_sec', 'word_count', 'split', 'keypoint_path', 'n_frames']

Duration statistics:
  Total duration: 2008.04 seconds (0.56 hours)
  Mean duration: 4.69 seconds
  Median duration: 4.67 seconds
  Min duration: 1.33 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 4,631
  Mean words per example: 10.82
  Median words: 10
  Min words: 1
  Max words: 31

Text statistics:
  Mean text length: 59.47 characters
  Median text length: 56 characters

============================================================
COMBINING TEST SPLIT
============================================================

How2Sign test: 1,464 examples
OpenASL test: 428 examples
Combined total: 1,892 examples

============================================================
COMBINED TEST (AFTER MERGING)
============================================================
Number of examples: 1,892
Columns: ['vid', 'text', 'start', 'end', 'duration_sec', 'word_count']

Duration statistics:
  Total duration: 8296.26 seconds (2.30 hours)
  Mean duration: 4.38 seconds
  Median duration: 4.22 seconds
  Min duration: 1.31 seconds
  Max duration: 8.00 seconds

Word count statistics:
  Total words: 23,523
  Mean words per example: 12.43
  Median words: 12
  Min words: 1
  Max words: 43

Text statistics:
  Mean text length: 64.02 characters
  Median text length: 59 characters

✓ Saved to: C:\My Projects\sign-language-bridge\data\final_full_test_dataset.tsv


████████████████████████████████████████████████████████████
FINAL SUMMARY
████████████████████████████████████████████████████████████

Split           How2Sign     OpenASL      Combined    
------------------------------------------------------------
Train           19,935       39,331       59,266      
Validation      1,051        461          1,512       
Test            1,464        428          1,892       
------------------------------------------------------------
TOTAL           22,450       40,220       62,670      

Total dataset duration: 276063.01 seconds (76.68 hours)

============================================================
✓ DATASET COMBINATION COMPLETE!
============================================================

Output files:
  - C:\My Projects\sign-language-bridge\data\final_full_train_dataset.tsv
  - C:\My Projects\sign-language-bridge\data\final_full_val_dataset.tsv
  - C:\My Projects\sign-language-bridge\data\final_full_test_dataset.tsv

Final schema: vid, text, start, end, duration_sec, word_count


SDPA attention
8-bit AdamW
grad checkpointing
Bucket batching