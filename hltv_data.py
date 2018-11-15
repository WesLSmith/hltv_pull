import cfscrape
from bs4 import BeautifulSoup as bs
import datetime as dt
from dateutil.relativedelta import relativedelta
import re
import pandas as pd
import numpy as np


def getTeams():
    scraper = cfscrape.create_scraper()
    teams_page = scraper.get("https://www.hltv.org/stats/teams").content
    teams_html = bs(teams_page, "html.parser")
    teams_html
    team_row = teams_html.findAll("td", class_= "teamCol-teams-overview")

    url_list = []
    for i in team_row:
        team_name = i.find("a")
        url = team_name.attrs['href']
        url_list.append("https://www.hltv.org/" + url)

    return url_list

def get3MonthCoreStats(team_url):
    today = dt.datetime.today().strftime('%Y-%m-%d')
    past = (dt.datetime.today() - relativedelta(months = 3)).strftime('%Y-%m-%d')

    team_url = team_url + f"?StartDate={past}&endDate={today}"
    team_page = scraper.get(team_url).content
    team_html = bs(team_page, 'html.parser')

    nameOfTeam = team_html.find("span", class_ = "context-item-name")
    nameOfTeam = nameOfTeam.text

    stats = team_html.findAll("div", class_="col standard-box big-padding")

    val_list = [i.text for i in stats]

    stats_dict = {}
    stats_dict["Name"] = nameOfTeam
    stats_dict["url"] = team_url
    stats_dict["maps_played"] = int(re.findall(r'\d+', val_list[0])[0])
    if stats_dict["maps_played"] == 0:
        return
    stats_dict["wdl"] = [int(i) for i in re.findall(r'\d+', val_list[1])]
    stats_dict["rounds_played"] =  int(re.findall(r'\d+', val_list[4])[0])
    try:
        stats_dict["roundsPerMap"] = stats_dict["rounds_played"]/stats_dict["maps_played"]
    except ZeroDivisionError:
        stats_dict["roundsPerMap"] = np.nan

    stats_dict["kdr"] = float(re.findall(r'\d+\.?\d*', val_list[5])[0])

    return stats_dict

def pastResults(team_url):
    match_history = team_url[:34] + "matches/" + team_url[34:]
    match_history = scraper.get(match_history).content
    match_html = bs(match_history, 'html.parser')

    nameOfTeam = match_html.find("span", class_ = "context-item-name")
    nameOfTeam = nameOfTeam.text

    matches = match_html.findAll("tr", class_ = 'group-1')
    match_dict_list = []
    for match in matches:
        match_dict = {}
        match_dict["team"] = nameOfTeam
        match_dict["date"] = match.find("td", class_ = "time").text
        match_dict["map"] = match.find("td", class_ = "statsMapPlayed").text
        match_dict["score"] = match.find('span', class_ = 'statsDetail').text
        try:
            match_dict["win"] = match.find('td', class_ = 'match-won').text
        except:
            try:
                match_dict["win"] = match.find('td', class_ = 'match-lost').text
            except:
                match_dict["win"] = match.find('td', class_ = 'match-tied').text
        match_dict['opponent'] = match.findAll('a')[3].text
        match_dict_list.append(match_dict)
    return match_dict_list

def cleanCoreStats(core_stats):
    core_stats["Win"] = core_stats["wdl"].apply(lambda x:x[0])
    core_stats["Tie"] = core_stats["wdl"].apply(lambda x:x[1])
    core_stats["Loss"] = core_stats["wdl"].apply(lambda x:x[2])
    core_stats = core_stats.drop(["wdl"], axis = 1)
    return core_stats

def cleanMatchStats(match_Stats):
    match_stats["roundsWon"] = match_stats["score"].apply(lambda x: int(x.split("-")[0]))
    match_stats["roundsLost"] = match_stats["score"].apply(lambda x: int(x.split("-")[2]))
    match_stats = match_stats.drop('score', axis = 1)
    return match_stats

if __name__ == "__main__":

    team_urls = getTeams()


    team_list = []
    match_list = []
    for i in team_urls:
        team_list.append(get3MonthCoreStats(i))
        match_list.append(pastResults(i))

    team_list = [i for i in dict_list if i is not None]

    df_list = []
    for d in match_list:
        df_list.append(pd.DataFrame(d))


    core_stats = pd.DataFrame(dict_list)
    match_stats = pd.concat(df_list)

    core_stats = cleanCoreStats(core_stats)
    match_stats = cleanMatchStats(match_stats)



    core_stats[core_stats["Name"] == "Kinguin"]

    core_stats[core_stats["Name"] == "Epsilon"]
