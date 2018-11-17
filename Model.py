import pandas as pd
import numpy as np
import statsmodels.api as sm



if __name__ == "__main__":
    team_data = pd.read_csv(r"C:\Users\Wesley Smith\Desktop\csgobetting\hltv_pull\teamstats.csv")
    match_stats = pd.read_csv(r"C:\Users\Wesley Smith\Desktop\csgobetting\hltv_pull\matchstats.csv")


    team_data
    team_data["W/L"] = team_data["Win"]/team_data["Loss"]

    match_stats["win"] = match_stats["win"].apply(lambda x: [1 if x == "W" else 0][0])
    match_stats["totalRounds"] = match_stats["roundsWon"] + match_stats["roundsLost"]


    joined_stats = pd.merge(match_stats, team_data, left_on = 'team', right_on="Name")
    joined_stats = pd.merge(joined_stats, team_data[["Name","W/L","roundsPerMap"]], left_on = "opponent", right_on = "Name")
    joined_stats

    exogenous = joined_stats[[
                            "W/L_x",
                            "W/L_y",
                            "kdr",
                            "maps_played",
                            "roundsPerMap_x",
                            "roundsPerMap_y"
                            ]]

    ols = sm.OLS(joined_stats["totalRounds"], exogenous)
    result = ols.fit()
    print(result.summary())

    test = joined_stats[joined_stats["team"] == "GOSU"]
    test2 = joined_stats[joined_stats["team"] == "CyberZen"]


    test["1"] = 26.524
    test["2"] = 1.805

    xnew = test[[
                "W/L_x",
                "2",
                # "W/L_y",
                "kdr",
                "maps_played",
                "roundsPerMap_x",
                # "roundsPerMap_y"
                "1"
                ]]
    pred = result.predict(xnew)
    pred
    test["pred"] = pred
    test["diff"] = test["totalRounds"] - test["pred"]
    test["diff"].mean()
