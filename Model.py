import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model

def cleanTeamData(team_data):

    team_data["W/L"] = team_data["Win"]/team_data["Loss"]
#test123
    return team_data[[
                    "Name",
                    "kdr",
                    "maps_played",
                    "roundsPerMap",
                    "rounds_played",
                    "W/L"
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
    team_data = pd.read_csv(r"C:\Users\wsmith\Desktop\personalDFS\csgo\hltv_pull\teamstats.csv")
    match_stats = pd.read_csv(r"C:\Users\wsmith\Desktop\personalDFS\csgo\hltv_pull\matchstats.csv")

    team_data

    team_data =cleanTeamData(team_data)
    match_stats = cleanMatchStats(match_stats)
    joined_stats = joinTeamMatches(team_data, match_stats)

    joined_stats

    exogenous = joined_stats[[
                            "kdr",
                            "kdr_opp",
                            "roundsPerMap",
                            "roundsPerMap_opp",
                            "W/L",
                            "W/L_opp",
                            "maps_played",
                            "maps_played_opp"
                            ]]


    train_features, test_features, train_labels, test_labels = train_test_split(exogenous, joined_stats["totalRounds"], test_size = 0.2)
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 5000, n_jobs = 3, max_depth=5,)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Mean Absolute % Accuracy:', round(accuracy, 2), '%.')




    test = joined_stats[joined_stats["Name"] == "Liquid"]
    test_op = joined_stats[joined_stats["Name"] == "Cloud9"]

    l3 = rf.predict(np.array([
            test["kdr"].mean(),
            test_op["kdr"].mean(),
            test["roundsPerMap"].mean(),
            test_op["roundsPerMap"].mean(),
            test["W/L"].mean(),
            test_op["W/L"].mean(),
            test["maps_played"].mean(),
            test_op["maps_played"].mean(),
             ]).reshape(1,-1))

    l3
