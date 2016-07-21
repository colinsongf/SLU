#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:25:33 2017

@author: shashank
"""

import requests
import json
import date_parser
import datetime

def connect_api(tags):

    return_output = {}
    fromloc=[]
    if 'fromloc' in tags:
        fromloc=tags['fromloc']

    toloc=[]
    if 'toloc' in tags:
        toloc=tags['toloc']

    round_trip=[]
    if 'round_trip' in tags:
        round_trip=tags['round_trip']

    airline_name=[]
    if 'airline_name' in tags:
        airline_name=tags['airline_name']
        airline_name=airline_name.replace(' ','_')

    airport_name=[]
    if 'airport_name' in tags:
        airport_name=tags['airport_name']
        airport_name=airport_name.replace(' ','_')

    depart_time=[]
    if 'depart_time' in tags:
       depart_time=tags['depart_time']

    depart_date=[]
    if 'depart_date' in tags:
        depart_date=tags['depart_date']

    return_date=[]
    if 'return_date' in tags:
        return_date=tags['return_date']

    if {'city_name','airport_code'}.issubset(tags) and 'fromloc' not in tags and 'toloc' not in tags:
        fromloc=tags['airport_code']
        toloc=tags['city_name']

    if {'airport_name','airport_code'}.issubset(tags) and 'fromloc' not in tags and 'toloc' not in tags:
        fromloc=tags['airport_code']
        toloc=tags['airport_name']

    elif {'city_name','toloc'}.issubset(tags) and 'fromloc' not in tags:
        fromloc=tags['city_name']

    elif {'city_name','fromloc'}.issubset(tags) and 'toloc' not in tags:
        toloc=tags['city_name']

    elif {'airport_name','toloc'}.issubset(tags) and 'fromloc' not in tags:
        fromloc=tags['airport_name']

    elif {'airport_name','fromloc'}.issubset(tags) and 'toloc' not in tags:
        toloc=tags['airport_name']

    elif {'airport_code','toloc'}.issubset(tags) and 'fromloc' not in tags:
        fromloc=tags['airport_code']

    elif {'airport_code','fromloc'}.issubset(tags) and 'toloc' not in tags:
        toloc=tags['airport_code']

    if 'fromloc' not in tags or 'toloc' not in tags:
        return {"error" : 'invalid query : no source or destination'}


    if 'depart_date' not in tags:
        depart_date="today"

    if 'depart_time' not in tags :
        depart_time=''

    (start_date,start_time)=date_parser.string_parse(depart_date)
    (_, start_time)=date_parser.string_parse(depart_date + ' ' + depart_time)

    end_date=''
    end_time=''
    if 'round_trip' in tags:
        if round_trip=='round trip':
            if 'return_date' not in tags:
                #let return date be one week from depart date
                temp_date=datetime.datetime.strptime(start_date, "%Y-%m-%d")
                return_date=temp_date + datetime.timedelta(days=7)
                return_date=str(return_date.date())
                del temp_date
                end_date=return_date
                end_time =''
            else:
                (end_date,end_time)=date_parser.string_parse(return_date)

    return_output['fromloc'] = fromloc
    return_output['toloc'] = toloc
    return_output['start_date'] = start_date
    return_output['start_time'] = start_time
    return_output['end_date'] = end_date
    return_output['end_time'] = end_time


    try :

        start_location_url="http://partners.api.skyscanner.net/apiservices/autosuggest/v1.0/US/USD/en-US?query="+fromloc+"&apiKey=de414398555714246923519339591707"
        if start_location_url.find(" ")!=-1:
            start_location_url=start_location_url.replace(" ","_")

        start_location_request=requests.get(start_location_url)
        start_location_json=json.loads(start_location_request.text)
        start_location_places=start_location_json['Places']
        start_location_place_id=start_location_places[0]['CityId'].encode('utf-8')
        del [start_location_url,start_location_request,start_location_json,start_location_places]

        return_output['start_location_id'] = start_location_place_id

        end_location_url="http://partners.api.skyscanner.net/apiservices/autosuggest/v1.0/US/USD/en-US?query="+toloc+"&apiKey=de414398555714246923519339591707"
        if end_location_url.find(" ")!=-1:
            end_location_url=end_location_url.replace(" ","_")

        end_location_request=requests.get(end_location_url)
        end_location_json=json.loads(end_location_request.text)
        end_location_places=end_location_json['Places']
        end_location_place_id=end_location_places[0]['CityId'].encode('utf-8')
        del [end_location_url,end_location_request,end_location_json,end_location_places]

        return_output['end_location_id'] = end_location_place_id
        response=requests.get("http://partners.api.skyscanner.net/apiservices/browseroutes/v1.0/US/USD/en-US/"+start_location_place_id+"/"+end_location_place_id+"/"+start_date+"/"+end_date+"?apiKey=de414398555714246923519339591707")
        response_json=response.json()
        quotes=response_json['Quotes']

        carriers=response_json['Carriers']
        carriers1={}

        for c in carriers :
            carriers1[c['CarrierId']] = c['Name']

        places=response_json['Places']
        places1 = {}
        for p in places :
            places1[p['PlaceId']] = p['Name']

        quotes1 = []
        for q in quotes :
            new_q = {'Price' : q['MinPrice']}
            if 'OutboundLeg' in q :
                outbound = q['OutboundLeg']
                carr = [carriers1[c] for c in outbound['CarrierIds']]
                dest = places1[outbound['DestinationId']]
                src = places1[outbound['OriginId']]

                outleg = {'Carriers' : carr, 'src' : src, 'dest' : dest, 'depdate' : outbound['DepartureDate']}
                new_q['outleg'] = outleg

            if 'InboundLeg' in q :
                outbound = q['InboundLeg']
                carr = [carriers1[c] for c in outbound['CarrierIds']]
                dest = places1[outbound['DestinationId']]
                src = places1[outbound['OriginId']]
                inleg = {'Carriers' : carr, 'src' : src, 'dest' : dest, 'depdate' : outbound['DepartureDate']}
                new_q['inleg'] = inleg

            quotes1.append(new_q)

        return_output['quotes'] = quotes1
    except :
        return_output['error'] = 'connection error'

    return return_output