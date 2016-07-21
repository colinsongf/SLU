import duckling

d = duckling.DucklingWrapper()
def parse_date_time(str):
	store=d.parse_time(str)
	return store

def string_parse(str):
	store=parse_date_time(str)
	dim=store[0]['dim'].encode('utf-8')
	assert dim=='time'
	del dim
	value=store[0]['value']

	if isinstance(value['value'],unicode):
		temp_str=value['value'].encode('utf-8')
		pivot1=temp_str.index('T')
		pivot2=temp_str.index('+')
		start_date=temp_str[0:pivot1]
		start_time=temp_str[pivot1+1:pivot2]
		end_date=start_date
		end_time='23:59:59.999'
		gmt=temp_str[pivot2+1:]
		del [temp_str,pivot1,pivot2]

	else:
		temp_str_from=value['value']['from'].encode('utf-8')
		temp_str_to=value['value']['to'].encode('utf-8')
		pivot1=temp_str_from.index('T')
		pivot2=temp_str_from.index('+')
		start_date=temp_str_from[0:pivot1]
		start_time=temp_str_from[pivot1+1:pivot2]

		pivot1=temp_str_to.index('T')
		pivot2=temp_str_to.index('+')
		end_date=temp_str_to[0:pivot1]
		end_time='23:59:59.999'
		gmt=temp_str_to[pivot2+1:]
		del [temp_str_to, temp_str_from,pivot1,pivot2]

	return (start_date,start_time)
