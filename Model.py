import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn import linear_model
from scipy.stats import randint as sp_randint

def cleanTeamData(team_data):

    team_data["W/L"] = team_data["Win"]/team_data["Loss"]

    return team_data[[
                    "Name",
                    "kdr",
                    "maps_played",
                    "roundsPerMap",
                    "rounds_played",
                    "W/L",
                    "l3won",
                    "l3lost"
                    ]]


def cleanMatchStats(match_stats):

    match_stats["win"] = match_stats["win"].apply(lambda x: [1 if x == "W" else 0][0])
    match_stats["totalRounds"] = match_stats["roundsWon"] + match_stats["roundsLost"]
    match_stats = match_stats[[
                "date",
                "team",
                "opponent",
                "map",
                "win",
                "totalRounds"
                ]]

    return match_stats

def joinTeamMatches(team_data, match_stats):

    joined_df = pd.merge(match_stats, team_data, left_on = 'team', right_on="Name")
    joined_df = pd.merge(joined_df, team_data, left_on = 'opponent', right_on="Name", suffixes = ["", "_opp"])
    return joined_df

if __name__ == "__main__":
    team_data = pd.read_csv(r"C:\Users\wsmith\Desktop\personalDFS\csgo\hltv_pull\teamstats.csv", encoding = 'latin-1' )
    match_stats = pd.read_csv(r"C:\Users\wsmith\Desktop\personalDFS\csgo\hltv_pull\matchstats.csv", encoding = 'latin-1' )

    team_data

    team_data =cleanTeamData(team_data)
    match_stats = cleanMatchStats(match_stats)
    joined_stats = joinTeamMatches(team_data, match_stats)

    exogenous = joined_stats[[
                            "kdr",
                            "kdr_opp",
                            "roundsPerMap",
                            "roundsPerMap_opp",
                            "W/L",
                            "W/L_opp",
                            "maps_played",
                            "maps_played_opp",
                            "l3won",
                            "l3won_opp",
                            "l3lost",
                            "l3lost_opp",

                            ]]
    joined_stats["binary"] = np.where(joined_stats["totalRounds"] > 26.5,1,0)

    train_features, test_features, train_labels, test_labels = train_test_split(exogenous, joined_stats["binary"], test_size = 0.15)
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators = 2500,
                               n_jobs = 7,
                               max_depth=5,
                               min_samples_leaf = .1
                               )

    param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 13),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

    opt_rf = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter = 50, cv =3)
    opt_rf.fit(train_features, train_labels)


    # Use the forest's predict method on the test data
    predictions = opt_rf.predict(test_features)

    ##metrics
    print(opt_rf.score(test_features,test_labels))



    test = joined_stats[joined_stats["Name"] == "Astralis"]
    test_op = joined_stats[joined_stats["Name"] == "Liquid"]
    len(test) > 0 and len(test_op) > 0

    l3 = opt_rf.predict_proba(np.array([
            test["kdr"].mean(),
            test_op["kdr"].mean(),
            test["roundsPerMap"].mean(),
            test_op["roundsPerMap"].mean(),
            test["W/L"].mean(),
            test_op["W/L"].mean(),
            test["maps_played"].mean(),
            test_op["maps_played"].mean(),
            test["l3won"].mean(),
            test_op["l3won"].mean(),
            test["l3lost"].mean(),
            test_op["l3lost"].mean()
            ,
             ]).reshape(1,-1))

    print(l3)
    opt_rf.classes_
