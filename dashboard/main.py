import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
import json
import pathlib
import logging
import time as tm
from scipy.signal import chirp, find_peaks, peak_widths
from django.core.serializers.json import DjangoJSONEncoder

log = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

def main():
    yesterday = datetime.now() - timedelta(hours=24*10)
    timeline = yesterday.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    filepath = collect_synth_time(timeline)

    couple = parse_events(filepath, 'Couple', 3, True, True, False, False, False)
    depro = parse_events(filepath, 'Deprotect', 3, True, False, True, False, False)
    flow_amino = parse_events(filepath, 'Amino Flow', 1, False, False, False, True, False)
    flow_activator = parse_events(filepath, 'Activator Flow', 1, False, False, False, True, False)
    flow_outlet = parse_events(filepath, 'Outlet Flow', 1, True, False, False, True, False)
    outlet_pressure = parse_events(filepath, 'Outlet Pressure', 1, True, False, False, False, False)


    if False:
        amino_couple = parse_by_amino(couple)
        amino_depro = parse_by_amino(depro)
        amino_outlet = parse_by_amino(flow_outlet)
        amino_flow_act = parse_by_amino(flow_activator)
        amino_flow_amino = parse_by_amino(flow_amino)

        couple_c, depro_c, activ_c, amino_c = read_control_data()
        render_series(flow_amino, 'amino valve flow rate', amino_c['mean'], amino_c['std'])
        render_series(flow_activator, 'activator valve flow rate', activ_c['mean'], activ_c['std'])
        render_ratio(flow_amino, flow_activator, 'Ratio of flow')
        render_amino(amino_flow_amino, 'amino flow rate per amino')
        render_amino(amino_flow_act, 'activator flow rate per amino')
        render_series(outlet_pressure, 'outlet pressure', False, False)
        render_series(flow_outlet, 'outlet flow', False, False)
        render_amino(amino_outlet, 'outlet flow')
        render_series(couple, 'Couple signal in time', couple_c['mean'], couple_c['std'])
        render_series(depro, 'Deprotect Signal in time', depro_c['mean'], depro_c['std'])
        render_amino(amino_couple, 'Couple per amino')
        render_amino(amino_depro, 'Deprotect per amino')


def read_control_data():
    with open('dashboard/store/control_values.json') as json_file:
        loaded = json.load(json_file)

    return loaded['coupling'], loaded['deprotect'], loaded['flow_activator'], loaded['flow_amino']

def mean_var(data):
    mean = np.mean(data)
    var = np.std(data)
    return mean, var


def normalize(data):
    data_mean = np.mean(data)
    out = [x/data_mean for x in data]
    return out


def render_outlet(flow, pressure):
    flow_normal = normalize(flow['value'])
    pressure_normal = normalize(pressure['value'])
    plt.figure(dpi=500)
    plt.title('Outlet')
    data = flow
    use_color = [f'#' + x[0:6] for x in data['uuid']]
    plt.scatter(range(0, len(pressure_normal)), pressure_normal, s=1, color='.7')
    plt.scatter(range(0, len(flow_normal)), flow_normal, s=1, color='r')
    plt.plot([0, len(flow_normal)], [1, 1], linewidth=1, color='k')
    plt.ylim(0, 3)
    plt.show()


def render_control_lines(mean, std, start, stop):
    plt.plot([start, stop], [mean, mean], linewidth=1, color='.1', alpha=.7)
    plt.plot([start, stop], [mean + std, mean + std], linewidth=1, color='g', alpha=.7)
    plt.plot([start, stop], [mean - std, mean - std], linewidth=1, color='g', alpha=.7)
    plt.plot([start, stop], [mean + 2 * std, mean + 2 * std], linewidth=1, color='orange', alpha=.7)
    plt.plot([start, stop], [mean - 2 * std, mean - 2 * std], linewidth=1, color='orange', alpha=.7)
    plt.plot([start, stop], [mean + 3 * std, mean + 3 * std], linewidth=1, color='r')
    plt.plot([start, stop], [mean - 3 * std, mean - 3 * std], linewidth=1, color='r')
    plt.text(start, mean+std*3+std/3, f'm: '+str(round(mean, 2)), fontsize=5)
    plt.text(start, mean+std*3, f's: '+str(round(std, 2)), fontsize=5)


def render_series(data, title, mean, std):
    plt.figure(dpi=200)
    plt.title(title)
    use_color = [f'#' + x[0:6] for x in data['uuid']]
    plt.scatter(range(0, len(data['value'])), data['value'], s=2, color=use_color)
    plt.plot(range(0, len(data['value'])), data['value'], linewidth=1, color='0.7', alpha=.3)
    if mean == False:
        mean, std = mean_var(data['value'])
    render_control_lines(mean, std, 0, len(data['value']))
    plt.ylim(bottom=0)
    plt.show()


def render_ratio(amino, activator, title):
    plt.figure(dpi=200)
    plt.title(title)
    use_color = [f'#' + x[0:6] for x in amino['uuid']]
    ratio_flow = np.divide(amino['value'], activator['value'])
    plt.scatter(range(0, len(ratio_flow)), ratio_flow, s=2, color=use_color)
    plt.plot(range(0, len(ratio_flow)), ratio_flow, linewidth=1, color='0.7', alpha=.3)
    plt.plot([0, len(ratio_flow)], [1, 1], color='.1', alpha=.4)
    plt.yscale("log")
    plt.show()


def render_amino(data, title):
    plt.figure(dpi=200)
    plt.title(title)
    s = 0
    for i in data:
        plt.scatter(range(s, s+len(data[i])), data[i], s=1, label=i)
        plt.plot(range(s, s+len(data[i])), data[i], linewidth=1, color='0.7', alpha=.3)
        mean, var = mean_var(data[i])
        render_control_lines(mean, var, s, s+len(data[i]))
        s = s+len(data[i])


    plt.legend(loc='lower left', fontsize='xx-small')
    plt.ylim(bottom=0)
    plt.show()


def parse_by_amino(data):
    amino_list = []
    for x in data['key']:
        if x not in amino_list:
            amino_list.append(x)
    amino_data = {}
    for j in amino_list:
        amino_data[j] = []
        for i, value in enumerate(data['key']):
            if value == j:
                amino_data[j].append(data['value'][i])
    return amino_data


def parse_events(file_path, event_name: str, event_length: int, baseline=False, skip_first_event=False,
                 skip_duplicates=False, absolute=False, plt_events=False):

    orig_name = os.path.basename(file_path)
    orig_path = os.path.dirname(file_path)
    path = pathlib.Path(orig_path)
    new_name = os.path.splitext(orig_name)[0]+event_name+f'.json'
    jsonpath = path / new_name
    # if os.path.isfile(jsonpath):
    #     log.info(f'Data cached: skipping parse '+event_name)
    #     with open(jsonpath) as json_file:
    #         loaded = json.load(json_file)
    #     return loaded

    with open(file_path) as json_file:
        loaded = json.load(json_file)
    key = {'value': [], 'time': [], 'uuid': [], 'key': [], 'step': [], 'color': [], 'index': []}
    print_event_name = event_name
    buffer = 0
    if event_name == 'Couple':
        data_type = 'uv'
        sub_ind = 'uv310'
    elif event_name == 'Deprotect':
        data_type = 'uv'
        sub_ind = 'uv275'
    elif event_name == 'Amino Flow':
        data_type = 'flow'
        sub_ind = 'amino'
        event_name = 'Deprotect Load'
    elif event_name == 'Activator Flow':
        data_type = 'flow'
        sub_ind = 'activator'
        event_name = 'Deprotect Load'
    elif event_name == 'Outlet Flow':
        data_type = 'flow'
        sub_ind = 'outlet'
        event_name = 'Couple'
        buffer = 5
    elif event_name == 'Outlet Pressure':
        data_type = 'pressure'
        sub_ind = 'pressure_out'
        event_name = 'Couple'
        buffer = 5
    index_counter=0
    for i in loaded:
        start_time = datetime.strptime(loaded[i]['machine']['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')
        skip_first = skip_first_event
        count_steps = 1
        for j, event in enumerate(loaded[i]['data']['event']['name']):
            if event == event_name:
                if skip_first:
                    skip_first = False
                    continue
                if skip_duplicates:
                    if loaded[i]['data']['event']['kwargs'][j]['id'] >= 2:
                        continue
                try:
                    event_time, event_amp = cut_out_sections(loaded[i]['data'], data_type, sub_ind, j, event_length, buffer)
                except IndexError:
                    continue
                if absolute:
                    event_amp = [abs(x) for x in event_amp]
                if baseline:
                    event_amp = event_amp - min(event_amp)


                # half peak outputs [0 width, 1 half-height, 2 peak_left, 3 peak_right]
                half_peak = peak_detection(event_amp)
                key['value'].append(half_peak[0] * half_peak[1])
                peak_center = (half_peak[2]+half_peak[3])/2
                key['time'].append(start_time + timedelta(seconds=peak_center+event_time[0]))
                key['uuid'].append(i)
                key['color'].append(f'#' + i[0:7])
                key['step'].append(count_steps)
                key['index'].append(index_counter)
                index_counter = index_counter + 1
                count_steps = count_steps + 1
                event_kwargs = loaded[i]['data']['event']['kwargs'][j]
                if 'ia' in event_kwargs.keys():
                    key['key'].append(loaded[i]['peptide']['aminos'][event_kwargs['ia']-1]['aa'])
                else:
                    key['key'].append([])

                if plt_events:
                    plt.plot(event_amp)
                    plt.hlines(*half_peak[1:], color="g")
                    plt.text(half_peak[0], half_peak[1], str(key['value'][-1]))
                    plt.show()
                    tm.sleep(.5)
    jsonpath.write_text(json.dumps(key, indent=1, cls=DjangoJSONEncoder))
    log.info(f'Parsed '+print_event_name)
    return key


def peak_detection(peak_amp):
    peaks, _ = find_peaks(peak_amp)
    half_peak = peak_widths(peak_amp, peaks, rel_height=0.5)
    sudo_area = np.multiply(half_peak[0], half_peak[1])
    half_width_ind = np.argmax(sudo_area)
    out = []
    for i in half_peak:
        out.append(i[half_width_ind])
    return out


def cut_out_sections(data, element, sub_id, event, event_length, buffer):
    a = find_nearest(data['time'], data['event']['time'][event])
    aind = data['time'].index(a)
    b = find_nearest(data['time'], data['event']['time'][event + event_length])
    bind = data['time'].index(b)

    cut_time = np.array(data['time'][aind-buffer:bind+buffer])
    cut_amp = np.array(data[element][sub_id][aind-buffer:bind+buffer])

    #remove empty data for future processing
    cut_time = cut_time[cut_time != np.array(None)]
    cut_amp = cut_amp[cut_amp != np.array(None)]
    return cut_time, cut_amp


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def collect_synth_time(timeline):
    # filename = f'health_synth'+timeline[0:11]+'.json'
    filename = f'health_synth.json'
    path = pathlib.Path('dashboard/store')
    jsonpath = path / filename

    # if os.path.isfile(jsonpath):
    #     log.info(f'Data cached: skipping collect')
    #     return jsonpath
    # Settings for connecting to mfg api
    #   Note the token is hardcoded for now
    url = 'https://mfg.mytidetx.com/api/v2'
    token = 'PEPTIDE-MAKER'
    headers = {'Authorization': token}
    metadata_url = url + f'/peptides/?created_after=' + timeline

    # Setup SSL for our internal certificate authority
    cacrt = 'Mytide Manufacturing CA.pem'
    response = requests.get(metadata_url, headers=headers, verify=cacrt)
    meta = response.json()  # each of these routes return JSON data

    j = 0
    collect_data = {}
    for i in meta:
        synth_url = url + f'/peptides/' + i['uuid'] + f'/synth/'
        synth = requests.get(synth_url, headers=headers, verify=cacrt)
        if synth.status_code == 200:
            synth_id = synth.json()[0]['synth_id']
            synth_request = requests.get(synth_url + synth_id + f'/', headers=headers, verify=cacrt)
            synth_data = synth_request.json()
            collect_data[i['uuid']] = synth_data
            j = j + 1

    # reorder the loaded json in time, oldest first/newest last
    start_time = []
    for jj in collect_data:
        start_time.append([jj, datetime.strptime(collect_data[jj]['machine']['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ')])
    sorted_start = sorted(enumerate(start_time), key=lambda x: x[1][1])
    new_order = [k[1][0] for k in sorted_start]
    reorder = {k: collect_data[k] for k in new_order}
    log.info(f'creating cache file: '+filename)
    jsonpath.write_text(json.dumps(reorder))
    return jsonpath


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s : %(asctime)s : %(name)s : %(message)s')
    log.info(f'Starting main function')
    main()
