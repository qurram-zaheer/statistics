import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina',
          'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan","Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State","RegionName"]  )'''
    with open('university_towns.txt') as file:
        data = []
        for line in file:
            data.append(line[:-1])
    state_town = []
    for line in data:
        if line[-6:] == '[edit]':
            state = line[:-6]
        elif '(' in line:
            town = line[:line.index('(')-1]
            state_town.append([state, town])
        # elif line[-1] == ':':
        #    town = line[:-1]
        #    state_town.append([state,town])
        # else:
        #    town = line[:line.index(',')]
        #    state_town.append([state,town])
        else:
            town = line
            state_town.append([state, town])
    state_college_df = pd.DataFrame(
        state_town, columns=['State', 'RegionName'])
    return state_college_df


get_list_of_university_towns()


def get_recession_start():
    df = pd.read_excel("gdplev.xls", skiprows=5).iloc[212:, 4:7]
    df.columns = ['Quarter', 'GDP in billions of current dollars',
                  'GDP in billions of chained 2009 dollars']

    for i in range(len(df)):
        if df.iloc[i, 1] > df.iloc[i+1, 1] and df.iloc[i+1, 1] > df.iloc[i+2, 1]:
            return df.iloc[i, 0]
    return("No recession in given time period")


get_recession_start()


def get_recession_end():
    df = pd.read_excel("gdplev.xls", skiprows=5).iloc[212:, 4:7]
    df.columns = ['Quarter', 'GDP in billions of current dollars',
                  'GDP in billions of chained 2009 dollars']
    df = df.set_index(df.columns[0])
    recession_start = list(df.index).index(get_recession_start())
    for i in range(36, len(df)):
        if df.iloc[i, 1] < df.iloc[i+1, 1] and df.iloc[i+1, 1] < df.iloc[i+2, 1]:
            return df.index[i + 2]
    return "No end to recession in given time period"


get_recession_end()


def get_recession_bottom():
    df = pd.read_excel("gdplev.xls", skiprows=5).iloc[212:, 4:7]
    df.columns = ['Quarter', 'GDP in billions of current dollars',
                  'GDP in billions of chained 2009 dollars']
    df = df.set_index(df.columns[0])
    recession_start = list(df.index).index(get_recession_start())
    recession_end = list(df.index).index(get_recession_end())
    argmin = df.iloc[recession_start:recession_end + 1, 1].argmin()
    return df.index[recession_start + argmin]


get_recession_bottom()


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''

    df = pd.read_csv("City_Zhvi_AllHomes.csv")
    qs = ['q1', 'q2', 'q3', 'q4']
    ind = list(df.columns).index('2000-01')
    year_raw = list(df.columns[ind:])
    quarters = []
    for item in year_raw:
        temp = item.split("-")
        for i in range(1, 5):
            quarters.append(temp[0] + 'q' + str(i))
    quarters = list(set(quarters))
    quarters.sort()

    for quarter in quarters:
        df[quarter] = df[df.columns[ind:ind+4]].mean(axis=1)

        ind = ind + 4
    temp = df['State'].apply(lambda x: states[x])
    df['State'] = temp
    df = df.sort('State')
    df = df.set_index(['State', 'RegionName'])

    return df.iloc[:, 249:-1]


convert_housing_data_to_quarters()
