'''
{'body_length': 4545,
 'channels': 8,
 'country': 'US',
 'currency': 'USD',
 'delivery_method': 1.0,
 'description': '<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;">\r\n<div class="im" style="color: #500050;">\r\n<p><span style="font-size: large; color: #000000;"><strong><span style="vertical-align: baseline; white-space: pre-wrap;">Please register here for the Winter 2013 Quarter of Shoreline Women\'s Midweek Study. &nbsp;</span></strong></span><br /><br /><span style="vertical-align: baseline; white-space: pre-wrap; color: #000000;">Winter 2013 Quarter is January-March.&nbsp;</span></p>\r\n</div>\r\n<p><span style="vertical-align: baseline; white-space: pre-wrap;">We will begin on Tuesday, January 8 and meet weekly through Tuesday, March 26 (no meeting February 19). &nbsp;</span></p>\r\n<p><span style="vertical-align: baseline; white-space: pre-wrap;">Please join us at the Mars Hill Shoreline building from 9:15 am &ndash; 11:15 am every Tuesday.</span><span style="font-size: 15.555556297302246px;">&nbsp;</span></p>\r\n<blockquote>\r\n<div>Jan 8 &nbsp;Week 1&nbsp;</div>\r\n<div>Jan 15 &nbsp;Week 2</div>\r\n<div>Jan 22 &nbsp;Week 3</div>\r\n<div>Jan 29 &nbsp;Week 4</div>\r\n<div>Feb 5 &nbsp;Week 5</div>\r\n<div>Feb 12 &nbsp;Week 6</div>\r\n<div>Feb 19 (no meeting)</div>\r\n<div>Feb 26 &nbsp;Week 7</div>\r\n<div>March 5 &nbsp; Week 8</div>\r\n<div>March 12 &nbsp;Week 9</div>\r\n<div>March 19 &nbsp;Week 10</div>\r\n<div>March 26 &nbsp;Week 11</div>\r\n</blockquote>\r\n<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;"><span style="vertical-align: baseline; white-space: pre-wrap;">We will be studying the New Testament book of Ephesians alongside Pastor Mark&rsquo;s sermon series entitled </span><span style="font-style: italic; vertical-align: baseline; white-space: pre-wrap;">Who Do You Think You Are?</span><span style="vertical-align: baseline; white-space: pre-wrap;"> We will be using an inductive study guide written just for you by a team of Mars Hill women. The guide is included in the registration fee. Please bring your own journal for personal notes. </span><span style="vertical-align: baseline; white-space: pre-wrap;">We&rsquo;re looking forward to digging into Ephesians together!</span></div>\r\n<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;"><span style="font-size: 15.555556297302246px;">&nbsp;</span></div>\r\n<div class="im" style="color: #500050;"><span style="color: #000000;"><strong style="font-size: 15.555556297302246px;"><span style="vertical-align: baseline; white-space: pre-wrap;">WMS Fall quarter cost:</span></strong></span><br /><span style="color: #000000;"><strong><span style="vertical-align: baseline; white-space: pre-wrap;">$45 with childcare</span></strong></span><br /><span style="color: #000000;"><strong><span style="vertical-align: baseline; white-space: pre-wrap;">$20 without childcare</span></strong></span></div>\r\n<div class="im" style="color: #500050;"><span style="color: #000000;"><strong><span style="vertical-align: baseline; white-space: pre-wrap;"><br /></span></strong></span></div>\r\n<div class="im" style="color: #500050;"><span style="vertical-align: baseline; white-space: pre-wrap; color: #000000;">**Women who are qualified to serve in MH Kids agree to serve with their table group on a rotating schedule. &nbsp;If you are not yet qualified to serve in MH Kids, we will work with you so that you can become qualified. &nbsp;Please let us know how we can help you.</span></div>\r\n<div class="im" style="color: #500050;"><span style="color: #222222; font-size: 15.555556297302246px;">&nbsp;</span></div>\r\n</div>\r\n<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;"><span style="white-space: pre-wrap;">WE ARE HIRING CHILDCARE STAFF! &nbsp;If you or anyone you know is interested in a paid child care position ($10 per hour), please contact Anie </span><span style="white-space: pre-wrap;">McDugle,</span>&nbsp;<span style="white-space: pre-wrap;"><a style="color: #1155cc;" href="mailto:aniemcdugle@gmail.com" target="_blank">aniemcdugle@gmail.com</a></span><span style="white-space: pre-wrap;">, ASAP!</span></div>\r\n<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;"><span style="white-space: pre-wrap;"><br /></span></div>\r\n<div style="color: #222222; font-family: Calibri, sans-serif; font-size: 15.555556297302246px;"><span style="white-space: pre-wrap;">Thank </span><span style="white-space: pre-wrap;">you!</span></div>\r\n<div><span style="white-space: pre-wrap;"><br /></span></div>',
 'email_domain': 'marshill.com',
 'event_created': 1353116126,
 'event_end': 1364338800,
 'event_published': 1353439856.0,
 'event_start': 1357678800,
 'fb_published': 0,
 'has_analytics': 0,
 'has_header': None,
 'has_logo': 0,
 'listed': 'y',
 'name': "MHC Shoreline | Winter Women's Midweek Study",
 'name_length': 44,
 'object_id': 4848389,
 'org_desc': '',
 'org_facebook': 11.0,
 'org_name': 'Mars Hill Church',
 'org_twitter': 12.0,
 'payee_name': '',
 'payout_type': 'ACH',
 'previous_payouts': []
 'sale_duration': 49.0,
 'show_map': 0,
 'ticket_types': [],
 'user_age': 212,
 'user_created': 1334773300,
 'user_type': 1,
 'venue_address': '',
 'venue_country': None,
 'venue_latitude': None,
 'venue_longitude': None,
 'venue_name': None,
 'venue_state': None,
 'sequence_number': 2940}


'''
def ticket_type(tt_entry):
    num_tiers = len(tt_entry)
    tt_id_list = []
    tt_cost_list = []
    tt_avail_list = []
    tt_quant_list = []
    for i in range(num_tiers):
        tt_id_list.append(tt_entry[i]['event_id'])
        tt_cost_list.append(tt_entry[i]['cost'])
        tt_avail_list.append(tt_entry[i]['availability'])
        tt_quant_list.append(tt_entry[i]['quantity_total'])
    return (num_tiers, tt_id_list, tt_cost_list, tt_avail_list, tt_quant_list)

def previous_payouts(pp_entry):
    num_payouts = len(pp_entry)
    name_list = []
    created_list = []
    country_list = []
    amount_list = []
    state_list = []
    address_list = []
    uid_list = []
    event_list = []
    zip_list = []

    for i in range(num_payouts):
        name_list.append(pp_entry[i]['name'])
        created_list.append(pp_entry[i]['created'])
        country_list.append(pp_entry[i]['country'])
        amount_list.append(pp_entry[i]['amount'])
        state_list.append(pp_entry[i]['state'])
        address_list.append(pp_entry[i]['address'])
        uid_list.append(pp_entry[i]['uid'])
        event_list.append(pp_entry[i]['event'])
        zip_list.append(pp_entry[i]['zip_code'])
    return (num_payouts, name_list, created_list, country_list, amount_list, state_list, address_list, uid_list, event_list, zip_list)


def scrub_words(text):
    """Basic cleaning of texts."""
    import nltk
    import string
    from nltk.tokenize import word_tokenize
    import unicodedata
    import re

    # remove html markup
    text=re.sub("(<.*?>)","",text)
    text=re.sub("&rsquo;","'",text)
    text=re.sub("&nbsp;","",text)
    text=re.sub("ndash","",text)
    text=re.sub("\r\n"," ",text)

    #remove non-ascii and digits
    ### for nlp may want to remove all punctuation
    #text=re.sub("(\\W|\\d)"," ",text)

    #remove whitespace
    text=text.strip()
  
    return text

def preprocess(row):
    
    live_entry = {}

    # clean up html code from description field
    live_entry['description'] = scrub_words(row['description'])

    # process ticket_type from 1 column with values as list of dictionaries 
    # to 5 columns with values as lists

    a, b, c, d, f = ticket_type(row['ticket_types'])
    
    live_entry['tt_num_tiers'] = a
    live_entry['tt_event_id'] = b
    live_entry['tt_cost'] = c
    live_entry['tt_avail'] = d
    live_entry['tt_quant'] = f

    # process previous_payouts from 1 column with values as list of dictionaries
    # to 9 columns with values as lists

    g, h, i, j, k, l, m, n, o, p = (live_entry['previous_payouts'])
    live_entry['pp_num_payouts'] = g
    live_entry['pp_name'] = h
    live_entry['pp_created'] = i
    live_entry['pp_country'] = j
    live_entry['pp_amount'] = k
    live_entry['pp_state'] = l
    live_entry['pp_address'] = m
    live_entry['pp_uid'] = n
    live_entry['pp_event'] = o
    live_entry['pp_zip_code'] = p


    columns = row.keys()
