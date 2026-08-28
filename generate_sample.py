import pandas as pd

### Phase 3
original_file = 'data/finalized_dynamic_supply_chain_logistics.csv'
df = pd.read_csv(original_file)

# Isolate the population (Rows 101 to 32066)
df_population = df.iloc[100:]

# Sample 1000 rows with a fixed seed for reproducibility
RANDOM_SEED = 42
df_sample = df_population.sample(n=1000, random_state=RANDOM_SEED)

# Sort for chronological consistency
df_sample = df_sample.sort_index()

# Save with Original Row Index for reviewer traceability
output_file = 'Phase3/phase3.csv'
df_sample.to_csv(output_file, index=True, index_label="Original_Row_Index")

print(f"Successfully generated {output_file} with 1000 rows.")
print(f"The random seed used was: {RANDOM_SEED}")


### Phase 2
# original_file = 'data/finalized_dynamic_supply_chain_logistics.csv'
# df = pd.read_csv(original_file)

# # Isolate the population you want to sample from (Rows 101 to 32066)
# # Pandas is 0-indexed, so row 101 is index 100.
# df_population = df.iloc[100:]

# # Sample exactly 100 rows randomly using a SEED (For Reproducibility)
# RANDOM_SEED = 42
# df_sample = df_population.sample(n=100, random_state=RANDOM_SEED)

# # Sort the index so the rows are in chronological order
# df_sample = df_sample.sort_index()

# # We keep index=True so the original row numbers are preserved in the new file
# output_file = 'Phase2/phase2.csv'
# df_sample.to_csv(output_file, index=True, index_label="Original_Row_Index")

# print(f"Successfully generated {output_file} with 100 rows.")
# print(f"The random seed used was: {RANDOM_SEED}")