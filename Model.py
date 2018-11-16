import pandas as pd
import numpy as np
import statsmodels.api as sm



if __name__ == "__main__":
    team_data = pd.read_csv(r"S:\Analytics\Wes_S\csgo HTLV\teamstats.csv")
    match_stats = pd.read_csv(r"S:\Analytics\Wes_S\csgo HTLV\matchstats.csv")


    team_data

    match_stats["win"] = match_stats["win"].apply(lambda x: [1 if x == "W" else 0][0])

    joined_stats = pd.merge(match_stats, team_data, left_on = 'team', right_on="Name")

    joined_stats

    exogenous = joined_stats[[
                            "Win",
                            "Loss",
                            "kdr",
                            "maps_played",
                            "roundsPerMap"
                            ]]

    logit = sm.Logit(joined_stats["win"], exogenous)
    result = logit.fit()
    print(result.summary())

    joined_stats[joined_stats["team"] == "Fragsters"]
    xnew = [54,
            55,
            .97,
            109,
            26.082569]
    result.predict(xnew)
