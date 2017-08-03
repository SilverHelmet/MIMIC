import scripts_util
from util import result_dir
import os
import json

def add_rank_attr(events, rank_key, rank_name):
    values = sorted([e[rank_key] for e in events], reverse = True)
    for e in events:
        e[rank_name] = values.index(e[rank_key]) + 1



def process(top_evnt_filepath, out_dir):
    events = []
    nb_patients = 46520
    for line in file(top_evnt_filepath):
        event = json.loads(line)
        if event['coverage'] == .0:
            event['frquency per patient'] = 0
        else:
            event['frequency per patient'] = (event['frequency'] + .0) / (nb_patients * event['coverage'])
        events.append(event)
    add_rank_attr(events, 'frequency', 'frequency-rank-all')
    add_rank_attr(events, 'coverage', 'coverage-rank-all')

    table2events = {}
    for e in events:
        table = e['table']
        if not table in table2events:
            table2events[table] = []
        table2events[table].append(e)
    
    for events in table2events.values():
        add_rank_attr(events, 'frequency', 'frequency-rank-table')
        add_rank_attr(events, 'coverage', 'coverage-rank-table')
    
    for table, events in table2events.iteritems():
        outf = file(os.path.join(out_dir, 'top_events_in_%s' %table), 'w')
        for e in sorted(events, key = lambda x: x['frequency'], reverse = True):
            outf.write(json.dumps(e) + "\n")
        outf.close()

    

    


if __name__ == "__main__":
    in_path = os.path.join(result_dir, 'top_freq_event.json')
    out_dir = os.path.join(result_dir, 'top_event')
    process(in_path, out_dir) 