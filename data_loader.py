import pandas as pd

#########################
# load benchmark data
#########################

# internal benchmark data
epoch_data = pd.read_csv("data/benchmarks_runs.csv")[["task", "model", "Best score (across scorers)"]]
epoch_data = epoch_data.rename(columns={"task": "benchmark", "Best score (across scorers)": "performance"})
epoch_data["source"] = "Epoch evaluations"

df_gpqa = epoch_data[epoch_data["benchmark"] == "GPQA diamond"]
df_fm_private = epoch_data[epoch_data["benchmark"] == "FrontierMath-2025-02-28-Private"]
df_math = epoch_data[epoch_data["benchmark"] == "MATH level 5"]
df_aime = epoch_data[epoch_data["benchmark"] == "OTIS Mock AIME 2024-2025"]
df_swebenchver = epoch_data[epoch_data["benchmark"] == "SWE-Bench verified"]

# external benchmark data
df_aider = pd.read_csv("data/external_benchmark_aider_polyglot.csv")[["Model version", "Percent correct", "Source"]]
df_aider = df_aider.rename(columns={"Model version": "model", "Percent correct": "performance", "Source": "source"})
df_aider.dropna(inplace=True)
df_aider["benchmark"] = "Aider polyglot"
df_aider["performance"] = df_aider["performance"] / 100 # convert to fraction

df_balrog = pd.read_csv("data/external_benchmark_balrog.csv")[["Model version", "Average progress"]]
df_balrog = df_balrog.rename(columns={"Model version": "model", "Average progress": "performance", "Source": "source"})
df_balrog.dropna(inplace=True)
df_balrog["benchmark"] = "Balrog"
df_balrog["performance"] = pd.to_numeric(df_balrog["performance"].str.rstrip('%'), errors="raise") / 100

df_factorio = pd.read_csv("data/external_benchmark_factorio_learning_environment.csv")[["Model version", "Lab Success %", "Source"]]
df_factorio = df_factorio.rename(columns={"Model version": "model", "Lab Success %": "performance", "Source": "source"})
df_factorio.dropna(inplace=True)
df_factorio["benchmark"] = "Factorio learning environment"
df_factorio["performance"] = pd.to_numeric(df_factorio["performance"].str.rstrip('%'), errors="raise") / 100

df_osworld = pd.read_csv("data/external_benchmark_os_world.csv")[["Model version", "Score", "Source"]]
df_osworld = df_osworld.rename(columns={"Model version": "model", "Score": "performance", "Source": "source"})
df_osworld.dropna(inplace=True)
df_osworld["benchmark"] = "OSWorld"
df_osworld["performance"] = pd.to_numeric(df_osworld["performance"], errors="raise") / 100

df_weirdml = pd.read_csv("data/external_benchmark_weirdml.csv")[["Model version", "Average", "Source"]]
df_weirdml = df_weirdml.rename(columns={"Model version": "model", "Average": "performance", "Source": "source"})
df_weirdml.dropna(inplace=True)
df_weirdml["benchmark"] = "WeirdML"
df_weirdml["performance"] = pd.to_numeric(df_weirdml["performance"].str.rstrip('%'), errors="raise") / 100

df_trivia = pd.read_csv("data/external_benchmark_triviaqa.csv")[["Model version", "EM", "Source"]]
df_trivia = df_trivia.rename(columns={"Model version": "model", "EM": "performance", "Source": "source"})
df_trivia.dropna(inplace=True)
df_trivia["benchmark"] = "TriviaQA"
df_trivia["performance"] = pd.to_numeric(df_trivia["performance"].str.rstrip('%'), errors="raise") / 100

df_openbook = pd.read_csv("data/external_benchmark_openbookqa.csv")[["Model version", "Accuracy", "Source"]]
df_openbook = df_openbook.rename(columns={"Model version": "model", "Accuracy": "performance", "Source": "source"})
df_openbook.dropna(inplace=True)
df_openbook["benchmark"] = "OpenBookQA"
df_openbook["performance"] = pd.to_numeric(df_openbook["performance"].str.rstrip('%'), errors="raise") / 100

df_hellaswag = pd.read_csv("data/external_benchmark_hellaswag.csv")[["Model version", "Overall accuracy", "Source"]]
df_hellaswag = df_hellaswag.rename(columns={"Model version": "model", "Overall accuracy": "performance", "Source": "source"})
df_hellaswag.dropna(inplace=True)
df_hellaswag["benchmark"] = "HellaSwag"
df_hellaswag["performance"] = pd.to_numeric(df_hellaswag["performance"].str.rstrip('%'), errors="raise") / 100

df_mmlu = pd.read_csv("data/external_benchmark_mmlu.csv")[["Model version", "EM", "Source"]]
df_mmlu = df_mmlu.rename(columns={"Model version": "model", "EM": "performance", "Source": "source"})
df_mmlu.dropna(inplace=True)
df_mmlu["benchmark"] = "MMLU"
df_mmlu["performance"] = pd.to_numeric(df_mmlu["performance"].str.rstrip('%'), errors="raise") / 100

df_winogrande = pd.read_csv("data/external_benchmark_winogrande.csv")[["Model version", "Accuracy", "Source"]]
df_winogrande = df_winogrande.rename(columns={"Model version": "model", "Accuracy": "performance", "Source": "source"})
df_winogrande.dropna(inplace=True)
df_winogrande["benchmark"] = "Winogrande"
df_winogrande["performance"] = pd.to_numeric(df_winogrande["performance"].str.rstrip('%'), errors="raise") / 100

df_bbh = pd.read_csv("data/external_benchmark_bbh.csv")[["Model version", "Average", "Source"]]
df_bbh = df_bbh.rename(columns={"Model version": "model", "Average": "performance", "Source": "source"})
df_bbh.dropna(inplace=True)
df_bbh["benchmark"] = "BBH"
df_bbh["performance"] = pd.to_numeric(df_bbh["performance"].str.rstrip('%'), errors="raise") / 100

df_vpct = pd.read_csv("data/external_benchmark_vpct.csv")[["Model version", "Correct", "Source"]]
df_vpct = df_vpct.rename(columns={"Model version": "model", "Correct": "performance", "Source": "source"})
df_vpct.dropna(inplace=True)
df_vpct["benchmark"] = "VPCT"
df_vpct["performance"] = pd.to_numeric(df_vpct["performance"].str.rstrip('%'), errors="raise") / 100

df_simple = pd.read_csv("data/external_benchmark_simple_bench.csv")[["Model version", "Score (AVG@5)", "Source"]]
df_simple = df_simple.rename(columns={"Model version": "model", "Score (AVG@5)": "performance", "Source": "source"})
df_simple.dropna(inplace=True)
df_simple["benchmark"] = "SimpleBench"
df_simple["performance"] = pd.to_numeric(df_simple["performance"].str.rstrip('%'), errors="raise") / 100

df_arc = pd.read_csv("data/external_benchmark_arc_agi.csv")[["Model version", "Score", "Source"]]
df_arc = df_arc.rename(columns={"Model version": "model", "Score": "performance", "Source": "source"})
df_arc.dropna(inplace=True)
df_arc["benchmark"] = "ARC-AGI"
df_arc["performance"] = pd.to_numeric(df_arc["performance"].str.rstrip('%'), errors="raise") / 100

df_cadeval = pd.read_csv("data/external_benchmark_cadeval.csv")[["Model version", "Overall pass (%)", "Source"]]
df_cadeval = df_cadeval.rename(columns={"Model version": "model", "Overall pass (%)": "performance", "Source": "source"})
df_cadeval.dropna(inplace=True)
df_cadeval["benchmark"] = "CadEval"
df_cadeval["performance"] = pd.to_numeric(df_cadeval["performance"].str.rstrip('%'), errors="raise") / 100

df_cybench = pd.read_csv("data/external_benchmark_cybench.csv")[["Model version", "Unguided % Solved", "Source"]]
df_cybench = df_cybench.rename(columns={"Model version": "model", "Unguided % Solved": "performance", "Source": "source"})
df_cybench.dropna(inplace=True)
df_cybench["benchmark"] = "Cybench"
df_cybench["performance"] = pd.to_numeric(df_cybench["performance"].str.rstrip('%'), errors="raise") / 100

df_deepresearch = pd.read_csv("data/external_benchmark_deepresearch.csv")[["Model version", "Average score", "Source"]]
df_deepresearch = df_deepresearch.rename(columns={"Model version": "model", "Average score": "performance", "Source": "source"})
df_deepresearch.dropna(inplace=True)
df_deepresearch["benchmark"] = "DeepResearch Bench"
df_deepresearch["performance"] = pd.to_numeric(df_deepresearch["performance"].str.rstrip('%'), errors="raise") / 100

# pd.read_csv("data/external_benchmark_fictionlivebench.csv") <-- which performance metric?

df_geobench = pd.read_csv("data/external_benchmark_geobench.csv")[["Model version", "ACW Country %", "Source"]]
df_geobench = df_geobench.rename(columns={"Model version": "model", "ACW Country %": "performance", "Source": "source"})
df_geobench.dropna(inplace=True)
df_geobench["benchmark"] = "GeoBench"
df_geobench["performance"] = pd.to_numeric(df_geobench["performance"].str.rstrip('%'), errors="raise") / 100

df_gsobench = pd.read_csv("data/external_benchmark_gso_bench.csv")[["Model version", "Score OPT@1", "Source"]]
df_gsobench = df_gsobench.rename(columns={"Model version": "model", "Score OPT@1": "performance", "Source": "source"})
df_gsobench.dropna(inplace=True)
df_gsobench["benchmark"] = "GSO-Bench"
df_gsobench["performance"] = pd.to_numeric(df_gsobench["performance"].str.rstrip('%'), errors="raise") / 100

# MCBench commented out - arena winrate metric doesn't fit well with benchmark stitching approach
# df_mcbench = pd.read_csv("data/external_benchmark_mcbench.csv")[["Model version", "Win rate", "Source"]]
# df_mcbench = df_mcbench.rename(columns={"Model version": "model", "Win rate": "performance", "Source": "source"})
# df_mcbench.dropna(inplace=True)
# df_mcbench["benchmark"] = "MCBench"
# df_mcbench["performance"] = pd.to_numeric(df_mcbench["performance"].str.rstrip('%'), errors="raise") / 100

df_osuniverse = pd.read_csv("data/external_benchmark_osuniverse.csv")[["Model version", "Weighted Score", "Source"]]
df_osuniverse = df_osuniverse.rename(columns={"Model version": "model", "Weighted Score": "performance", "Source": "source"})
df_osuniverse.dropna(inplace=True)
df_osuniverse["benchmark"] = "OSUniverse"
df_osuniverse["performance"] = pd.to_numeric(df_osuniverse["performance"].str.rstrip('%'), errors="raise") / 100

df_terminal = pd.read_csv("data/external_benchmark_terminal_bench.csv")[["Model version", "Accuracy mean", "Source"]]
df_terminal = df_terminal.rename(columns={"Model version": "model", "Accuracy mean": "performance", "Source": "source"})
df_terminal.dropna(inplace=True)
df_terminal["benchmark"] = "Terminal Bench"
df_terminal["performance"] = pd.to_numeric(df_terminal["performance"].str.rstrip('%'), errors="raise") / 100

df_videomme = pd.read_csv("data/external_benchmark_videomme.csv")[["Model version", "Overall (no subtitles)", "Source"]]
df_videomme = df_videomme.rename(columns={"Model version": "model", "Overall (no subtitles)": "performance", "Source": "source"})
df_videomme.dropna(inplace=True)
df_videomme["benchmark"] = "VideoMME"
df_videomme["performance"] = pd.to_numeric(df_videomme["performance"].str.rstrip('%'), errors="raise") / 100

#########################
# data processing and filtering
#########################
# possible outliers
# SWE-bench verified: o4-mini and o3 do surprisingly badly because of function calling issues
models_to_drop = [
    'o3-mini-2025-01-31_medium',
    'o4-mini-2025-04-16_medium',
]
df_swebenchver = df_swebenchver[~df_swebenchver['model'].isin(models_to_drop)].reset_index(drop=True)

benchmarks = [
    df_gpqa, # https://www.anthropic.com/news/claude-3-5-sonnet specifically GPQA diamond
    df_fm_private, # not public. but probably OpenAI models have hill climbed on it to some extent. will say that it has been hill-climbed on
    df_math, # minerva https://arxiv.org/abs/2206.14858
    df_aime, # https://openai.com/index/introducing-o3-and-o4-mini/
    # df_swebenchver, # definitely optimised for
    df_aider, # https://openai.com/index/gpt-4-1/ so maybe?
    df_balrog, # who's even heard of this?
    df_factorio, # probably not for most
    df_osworld, # maybe? https://www.anthropic.com/news/3-5-models-and-computer-use
    df_weirdml, # probably not
    df_trivia, # GPT-3 https://arxiv.org/abs/2005.14165
    df_openbook, # GPT-3 https://arxiv.org/abs/2005.14165
    df_hellaswag, # https://cdn.openai.com/papers/gpt-4.pdf
    df_mmlu, # https://cdn.openai.com/papers/gpt-4.pdf
    df_winogrande, # https://cdn.openai.com/papers/gpt-4.pdf
    df_bbh, # chinchilla https://arxiv.org/abs/2203.15556
    df_vpct, # probably not
    df_simple, # probably not
    df_arc, # GPT-3 https://arxiv.org/abs/2005.14165
    df_cadeval, # never even heard of this, also seems pretty hard to find on google. idk how we even found out about this! https://willpatrick.xyz/cadevalresults_20250422_095709/
    df_cybench, # probably? e.g. claude 3.7 sonnet. https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf it's one of their RSP evals though, so maybe they have some disincentive to do well on this
    df_deepresearch, # really new, probably not
    df_geobench, # probably not?
    df_gsobench, # probably not, really new! https://x.com/slimshetty_/status/1932491280971608253
    # df_mcbench, # commented out - arena winrate metric doesn't fit benchmark stitching
    # df_osuniverse, # seems likely very sensitive to the specific agent scaffold, messes up the fit
    df_terminal, # mildly since very recent, e.g. 
    # df_videomme 
]

optim_bench = [
  df_fm_private,
  df_gpqa,
  df_math,
  df_aime,
  df_swebenchver,
  df_aider,
  df_osworld,
  df_trivia,
  df_openbook,
  df_hellaswag,
  df_mmlu,
  df_winogrande,
  df_bbh,
  df_arc,
  df_cybench,
  df_terminal, # https://www.anthropic.com/news/claude-4
  df_videomme # https://blog.google/products/gemini/gemini-2-5-pro-updates/ https://openai.com/index/gpt-4-1/
]

# benchmark release dates
sheet_id = "17YuRlWlFqfeqztj_s9VlfIruUOnGCpf5S2eMjC2Oz_Q"
gid = "0"  # the sheet’s gid
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
bench_dates = pd.read_csv(csv_url)

scores_df = pd.concat(benchmarks, ignore_index=True)
optim_names = {df['benchmark'].iloc[0] for df in optim_bench}
scores_df['optimized'] = scores_df['benchmark'].isin(optim_names)
raw_perf = scores_df['performance'].copy()
scores_df['performance'] = pd.to_numeric(scores_df['performance'], errors='coerce')
dropped = scores_df[ raw_perf.notna() & scores_df['performance'].isna() ]
print("null performances after coercion:", scores_df['performance'].isna().sum())

unique_benchmarks = scores_df['benchmark'].unique()
benchmark_to_id = {benchmark: f"b{i+1}" for i, benchmark in enumerate(unique_benchmarks)}

unique_models = scores_df['model'].unique()
model_to_id = {model: f"m{i+1}" for i, model in enumerate(unique_models)}

scores_df['benchmark_id'] = scores_df['benchmark'].map(benchmark_to_id)
scores_df['model_id'] = scores_df['model'].map(model_to_id)
scores_df = scores_df[['benchmark_id', 'benchmark', 'model_id', 'model', 'performance', 'optimized', 'source']]

saturation_level = 0 #0.025
scores_df = scores_df[(scores_df["performance"] >= saturation_level) & (scores_df["performance"] <= 1 - saturation_level)]
print("after saturation filter", len(scores_df))

# Count the number of benchmarks per model
model_benchmark_counts = scores_df.groupby('model')['benchmark'].nunique()

# Filter out models that are only evaluated on one benchmark
models_to_keep = model_benchmark_counts[model_benchmark_counts > 3].index # change number of benchmarks evaluated on, default 1
scores_df = scores_df[scores_df['model'].isin(models_to_keep)]
print("after filter num benchmarks", len(scores_df))

# After filtering, update the unique models and model_to_id mapping
unique_benchmarks = scores_df['benchmark'].unique()
benchmark_to_id = {benchmark: f"b{i+1}" for i, benchmark in enumerate(unique_benchmarks)}

# Reset the model IDs after filtering
unique_models = scores_df['model'].unique()
model_to_id = {model: f"m{i+1}" for i, model in enumerate(unique_models)}

# Apply the updated mappings
scores_df['benchmark_id'] = scores_df['benchmark'].map(benchmark_to_id)
scores_df['model_id'] = scores_df['model'].map(model_to_id)
scores_df = scores_df[['benchmark_id', 'benchmark', 'model_id', 'model', 'performance', 'optimized', 'source']]

df_model = pd.read_csv("data/model_versions.csv")[["id", "Model", "Version release date"]]
df_model = df_model.rename(columns={"id": "model", "Version release date": "date"})
df_model.loc[df_model["model"] == "gemini-exp-1206","date"] = "2024-12-06" # typo, in the benchmark it says 2025-12-06 which is impossible. to be updated.
scores_df = scores_df.merge(df_model, on="model")
print("after merge with model versions", len(scores_df))

scores_df = scores_df.merge(
    bench_dates[['benchmark', 'benchmark_release_date']],
    on='benchmark',
    how='left'
)
scores_df['benchmark_release_date'] = pd.to_datetime(scores_df['benchmark_release_date'])
print("after merge with benchmark dates", len(scores_df))

print(f"Original number of rows: {len(scores_df)}")
# Group by model and benchmark, then apply the aggregation.
# We use gmean for the performance score and keep the 'first' value for metadata columns.
scores_df_aggregated = scores_df.groupby(['model_id', 'benchmark_id']).agg({
    'performance': "min", #gmean, "max"
    'benchmark': 'first',
    'benchmark_release_date': 'first',
    'optimized': 'first',
    'model': 'first',
    'date': 'first',
    'source': 'first',
    # Add any other columns you want to keep here, using 'first' as the aggregator
}).reset_index()

print(f"Number of rows after aggregation: {len(scores_df_aggregated)}")

# Overwrite the original dataframe with the aggregated one
scores_df = scores_df_aggregated

if __name__ == "__main__":
    print(scores_df)