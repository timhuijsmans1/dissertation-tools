from datetime import datetime
from dateutil import parser
from pytz import utc
from pytz import timezone

eastern = timezone('US/Eastern')
datetime_string = '2021-07-21T19:55:29'
datetime_object = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')
aware_datetime = datetime_object.replace(tzinfo=eastern)
eastern_time = aware_datetime.astimezone(utc)
utc_again = eastern_time.astimezone(eastern)
print(aware_datetime)
print(eastern_time)
print(utc_again)

print('-' * 25)

eastern = timezone('US/Eastern')
datetime_string = '2021-07-21T23:55:29'
datetime_object = datetime.strptime(datetime_string, '%Y-%m-%dT%H:%M:%S')
aware_datetime = datetime_object.replace(tzinfo=utc)
eastern_time = aware_datetime.astimezone(eastern)
utc_again = eastern_time.astimezone(utc)
print(aware_datetime)
print(eastern_time)
print(utc_again)