"""
Download historical weather data using wunderground API. https://www.wunderground.com/weather/api/d/docs
This script is based on @jtelszasz work. https://github.com/jtelszasz/my_energy_data_viz

Basic Usage:
    python scrape_weather_data.py startMonth startDay startYear endMonth endDay endYear
    python scrape_weather_data.py 3 1 2017 3 31 2017

`key`: API key.
`zmw`: a city's zmw code, see https://www.wunderground.com/weather/api/d/docs?d=autocomplete-api&MR=1
`save_path`: download data file to this path.
`empty_obs_dict`: this dict structure will be reused a few times for storing data. https://www.wunderground.com/weather/api/d/docs?d=resources/phrase-glossary
    {
        'tempm': temp in C
        'wspdm': windspeed kph
        'precipm': precipitation in mm
        'conds': weather condition phrases
    }

"""
import sys
import pandas as pd
import numpy as np
import datetime
import urllib.request
import json
import copy
import time

key = '0f63b2474e401b5d'

zmw = '00000.1.59493'  # Shenzhen, China 深圳
# zmw = '00000.1.59082'  # Shaoguan, China 韶关
# this dict structure will be reused a few times
empty_obs_dict = {'tempm': [], 'hum': [],
                  'wspdm': [], 'precipm': [], 'conds': []}

save_path = 'D:\Study\mierda\data\load_forecast\weather_data\\'


def download_one_day_data(weather_date):
    '''
    Download one day's weather data.
    Return a parsed json.
    '''

    YYYYMMDD = weather_date.strftime('%Y%m%d')
    query = 'http://api.wunderground.com/api/%s/history_%s/q/zmw:%s.json' % (
        key, YYYYMMDD, zmw)
    f = urllib.request.urlopen(query)
    json_string = f.read()
    parsed_json = json.loads(json_string)
    f.close()
    prettydate = parsed_json['history']['date']['pretty']
    print(prettydate)
    return parsed_json


def parse_weather_data(parsed_json, input_dict):
    '''
    Returns:
        a timestamp list
        a dict using the structure of input_dict.
    Timestamp is corresponding to each observation.
    Each key contains the hourly observations for this day (a list).
    '''
    timestamp_list = []
    obs_dict = copy.deepcopy(input_dict)
    obs_num = len(parsed_json['history']['observations'])

    for i in range(obs_num):
        year = int(parsed_json['history']['observations'][i]['date']['year'])
        month = int(parsed_json['history']['observations'][i]['date']['mon'])
        day = int(parsed_json['history']['observations'][i]['date']['mday'])
        hour = int(parsed_json['history']['observations'][i]['date']['hour'])
        minute = int(parsed_json['history']['observations'][i]['date']['min'])
        timestamp_list.append(datetime.datetime(
            year, month, day, hour, minute))

        for obs in obs_dict:
            # obs are the features like temp, windspeed
            try:
                value = float(parsed_json['history']['observations'][i][obs])
            except:
                # weather conds are strings, can't be converted to float
                value = parsed_json['history']['observations'][i][obs]
            if value == -9999:
                value = np.nan
            obs_dict[obs].append(value)

    return timestamp_list, obs_dict


def main():
    if len(sys.argv) != 7:
        print('Not enough date args')
        sys.exit(1)
    startMonth = int(sys.argv[1])
    startDay = int(sys.argv[2])
    startYear = int(sys.argv[3])
    endMonth = int(sys.argv[4])
    endDay = int(sys.argv[5])
    endYear = int(sys.argv[6])

    startDate = datetime.datetime(
        year=startYear, month=startMonth, day=startDay)
    endDate = datetime.datetime(year=endYear, month=endMonth, day=endDay)

    if startDate > endDate:
        raise Exception('Invalid date arguments.')

    # store all days' obs data
    full_timestamp_list = []
    full_obs_dict = copy.deepcopy(empty_obs_dict)

    currentDate = startDate
    count = 0  # the API has a limit of 10 calls per minute

    while currentDate <= endDate:
        parsed_json = download_one_day_data(currentDate)
        daily_timestamp, daily_obs_dict = parse_weather_data(
            parsed_json, empty_obs_dict)

        # merge each day's data
        # don't use append -> [a, [b]]
        full_timestamp_list.extend(daily_timestamp)
        {full_obs_dict[obs].extend(daily_obs_dict[obs])
         for obs in full_obs_dict}

        currentDate += datetime.timedelta(days=1)
        count += 1
        if count == 10:
            count = 0
            print('Pause for 60 seconds...')
            time.sleep(60)

    df = pd.DataFrame(full_obs_dict, index=full_timestamp_list)
    start_string = startDate.strftime("%Y%m%d")
    end_string = endDate.strftime("%Y%m%d")
    file_path = save_path + 'weather_data_' + start_string + '-' + end_string
    df.to_csv('%s.csv' % file_path)


if __name__ == '__main__':
    main()
