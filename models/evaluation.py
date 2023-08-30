import json
import requests
import re
from datetime import date, timedelta, datetime
import html
import os
import shutil
import pickle

import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np
from sacrebleu import CHRF

sb_chrf = CHRF(CHRF.CHAR_ORDER, CHRF.WORD_ORDER, CHRF.BETA, lowercase=False, whitespace=False, eps_smoothing=False)


def get_exact_match(out_queries, ref_queries):
    exact_matches = list()
    total = 0
    for p, l in zip(out_queries, ref_queries):
        if normalize_query(p) == normalize_query(l):
            exact_matches.append(1)
        else:
            exact_matches.append(0)
        total += 1
    return np.mean(exact_matches) * 100, exact_matches


def normalize_query(s):
    s = re.sub(r'out:[a-z]+', 'out:json', s)
    s = re.sub(r'timeout:\d+', 'timeout:300', s)
    s = re.sub(r'\s+', '', s)  # remove all whitespace
    return s


def get_oqo_score(out_queries, ref_queries, converter, weights=[1, 1, 1]):
    assert len(ref_queries) == len(out_queries)
    assert len(weights) == 3

    ref_xml_queries = [get_overpass_xml(q, converter) for q in ref_queries]
    out_xml_queries = [get_overpass_xml(q, converter) for q in out_queries]

    assert len(ref_xml_queries) == len(ref_queries)
    assert len(out_xml_queries) == len(out_queries)
    assert len(ref_queries) == len(out_queries)
    assert len(ref_xml_queries) == len(out_xml_queries)
    scores = dict(kv_overlap=list(), xml_overlap=list(), chrf=list(), oqo=list())
    for i in range(len(ref_queries)):
        ref_xml, out_xml = ref_xml_queries[i], out_xml_queries[i]
        ref_query, out_query = ref_queries[i], out_queries[i]

        # order is important because get_xml_overlap_score modifies the xml tree
        kv_overlap = get_key_value_overlap_score(ref_xml, out_xml)
        xml_overlap = get_xml_overlap_score(ref_xml, out_xml)
        chrf = get_character_fscore(ref_query, out_query)

        combined = np.average([kv_overlap, xml_overlap, chrf], weights=weights)

        scores['kv_overlap'].append(kv_overlap)
        scores['xml_overlap'].append(xml_overlap)
        scores['chrf'].append(chrf)

        scores['oqo'].append(combined)

    converter.save_cache()

    score = {key: np.mean(values) * 100 for key, values in scores.items()}
    return score, scores


class OverpassXMLConverter:

    def __init__(self, url, cache_dir='cache', save_frequency=100):
        self.url = url
        self.pre_pattern = re.compile(r"<pre>(.*?)</pre>", flags=re.DOTALL)

        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, 'xml_converter_cache.json')
        self.cache = dict()
        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

        self.save_frequency = save_frequency
        self.api_calls = 0

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
        print('xml_converter_cache saved')

    def convert(self, query, timeout=20):

        if query in self.cache:
            return self.cache[query]

        url = self.url + '?data='
        url += requests.utils.quote(query)
        url += "&target=xml"

        response = requests.get(url, timeout=timeout)
        self.api_calls += 1
        text = response.text
        match = self.pre_pattern.search(text)
        if not match:
            raise ValueError(f'no match found in request output: {text}')
        text = match.group(1).strip()
        xml_query = html.unescape(text)

        self.cache[query] = xml_query
        if self.save_frequency > 0 and len(self.cache) % self.save_frequency == 0:
            self.save_cache()

        return xml_query


def get_overpass_xml(query, converter):
    try:
        query = replace_overpass_turbo_vars(query)
    except Exception:
        pass
    xml_query = converter.convert(query)
    try:
        tree = ET.fromstring(xml_query)
    except ET.ParseError as e:
        print('Could not parse into xml tree:')
        print(xml_query)
        tree = ET.fromstring('<osm-script output="json" output-config="" timeout="300"></osm-script>')
    return tree


def elements_equal(e1, e2):
    if e1.tag != e2.tag:
        return False
    if e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


def get_xml_overlap_score(ref_xml, out_xml):
    remove_attrib = ['into', 'from', 'timeout', 'k', 'v', 'regk', 'regv']

    ref_node_map = defaultdict(list)
    num_ref_subtrees = 0
    stack = [ref_xml]
    while stack:
        node = stack.pop()
        for attrib_name in remove_attrib:
            if attrib_name in node.attrib:
                del node.attrib[attrib_name]
        num_ref_subtrees += 1
        node_key = str(node.tag) + str(node.attrib) + str(node.text)
        ref_node_map[node_key].append(node)
        for child in node:
            stack.append(child)

    stack = [out_xml]
    while stack:
        node = stack.pop()
        for attrib_name in remove_attrib:
            if attrib_name in node.attrib:
                del node.attrib[attrib_name]
        for child in node:
            stack.append(child)

    num_equal_subtrees = 0
    stack = [out_xml]
    while stack:
        node = stack.pop()
        node_key = str(node.tag) + str(node.attrib) + str(node.text)
        for ref_subtree in ref_node_map[node_key][:]:
            is_equal = elements_equal(node, ref_subtree)
            if is_equal:
                num_equal_subtrees += 1
                ref_node_map[node_key].remove(ref_subtree)

        for child in node:
            stack.append(child)
    # print(num_equal_subtrees, num_ref_subtrees)
    return num_equal_subtrees / num_ref_subtrees


def get_key_value_overlap_score(ref_xml, out_xml):
    def get_key_values(xml_tree):
        keys_values = set()
        stack = [xml_tree]
        while stack:
            node = stack.pop()
            k = None
            v = None
            if 'k' in node.attrib:
                k = node.attrib['k']
            if 'regk' in node.attrib:
                k = node.attrib['regk']
            if 'v' in node.attrib:
                v = node.attrib['v']
            if 'regv' in node.attrib:
                v = node.attrib['regv']

            if k:
                keys_values.add(k)
            if v:
                keys_values.add(v)
            if k and v:
                keys_values.add((k, v))

            for child in node:
                stack.append(child)
        return keys_values

    ref_keys_values = get_key_values(ref_xml)
    out_keys_values = get_key_values(out_xml)

    if max(len(ref_keys_values), len(out_keys_values)) == 0:
        return 1.0

    num_overlap = len(ref_keys_values & out_keys_values)
    if num_overlap == 0:
        return 0.0

    return num_overlap / max(len(ref_keys_values), len(out_keys_values))


def get_character_fscore(ref_query, out_query):
    output = sb_chrf.corpus_score([out_query], [[ref_query]])
    return output.score / 100


class Overpass:

    def __init__(self, url, cache_dir='cache', cache_filename='overpass_soft_cache', save_frequency=10):
        self.url = url
        os.makedirs(cache_dir, exist_ok=True)

        self.cache_file = os.path.join(cache_dir, cache_filename + '.pickle')
        print('load cache from: ' + self.cache_file)
        self.cache = dict()
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)

        self.save_frequency = save_frequency
        self.cache_modified = 0

        self.num_url_requests = 0
        self.num_cache_hits = 0

    def save_cache(self):
        if self.cache_modified == 0:
            print('cache not saved')
            return

        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        self.cache_modified = 0

        # copy as backup if process killed while writing
        shutil.copyfile(self.cache_file, self.cache_file + '_ba')
        print('overpass_cache cache saved')

    def add_cache(self, key, value):
        if key in self.cache and self.cache[key] == value:
            return

        self.cache[key] = value
        self.cache_modified += 1

        if self.cache_modified % self.save_frequency == 0:
            self.save_cache()

    def query(self, query, use_cache=True, timeout=500):
        if use_cache and query in self.cache:
            print('read from cache')
            self.num_cache_hits += 1
            return tuple(self.cache[query])

        try:
            result = self._query(query, timeout)
        except requests.ConnectionError as e:
            print('connection error')
            raise e
        except requests.Timeout as e:
            result = ('server_error_timeout', -1)
        except Exception:
            result = ('unknown_error', -2)

        self.add_cache(query, result)
        return result

    def _query(self, query, timeout=500):

        self.num_url_requests += 1
        response = requests.get(self.url, params={"data": query}, timeout=timeout)
        text = response.text

        error = None
        if len(text) > 2000000000:
            error = 'memory_error'
        elif 'parse error:' in text:
            error = 'parse_error'
        elif 'static error: The selected output format does not support the diff or adiff mode.' in text:
            error = 'static_error_diff'
        elif 'static error:' in text and 'timestamp' in text:
            error = 'static_error_timestamp'
        elif 'static error:' in text and 'Invalid regular expression' in text:
            error = 'static_error_re'
        elif 'static error:' in text and 'Unknown attribute &quot;regexp&quot;' in text:
            error = 'static_error_re'
        elif 'runtime error:' in text and 'Query timed out in' in text:
            error = 'runtime_error_timeout'
        elif 'static error:' in text and 'A convert statement can have' in text:
            error = 'static_error_convert'
        elif 'static error:' in text and 'in bounding boxes must be between' in text:
            error = 'static_error_bbox'
        elif 'static error:' in text and 'bbox-query' in text:
            error = 'static_error_bbox'
        elif 'static error:' in text and 'the only allowed values are non-empty strings' in text:
            error = 'static_error_attribute_value'
        elif 'static error:' in text and 'For the attribute' in text:
            error = 'static_error_attribute_value'

        if error is not None:
            result = (error, -1)
            return result

        #print(text)
        j = json.loads(text)
        if 'remark' in j:
            print('remark', j['remark'])
        elements = j['elements']

        if len(elements) in [1, 2] and elements[0]['type'] == 'count':
            if 'total' in elements[0]['tags']:
                num = elements[0]['tags']['total']
            else:
                num = elements[0]['tags']['all']
            result = (f'count_{num}', num)
            return result

        types_ids = list()
        for element in elements:
            #assert element['type'] in {'way', 'node', 'relation', 'area', 'count', 'stat', 'elem', 'data', 'geometry', 'rel', 'out', 'info'}
            types_ids.append((element['type'] + '_' + str(element['id'])))
        types_ids = list(sorted(types_ids))
        result = (types_ids, len(types_ids))

        return result


class Nominatim:

    def __init__(self, url, cache_dir='cache', save_frequency=10):
        self.url = url

        self.cache_file = os.path.join(cache_dir, 'nominatim_cache.json')
        self.cache = dict()
        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

        self.save_frequency = save_frequency

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except KeyboardInterrupt:
            self.save_cache()
            exit()
        print('nominatim_cache saved')

    def query(self, text, centre=False, bbox=False, timeout=30):
        assert sum([centre, bbox]) <= 1
        text = text.strip().strip('"')

        cache_key = text
        if centre:
            cache_key += '_centre'
        if bbox:
            cache_key += '_bbox'

        if cache_key in self.cache:
            return self.cache[cache_key]

        params = {"q": text, "format": "jsonv2", "limit": "1"}
        response = requests.get(self.url, params=params, timeout=timeout)
        results = json.loads(response.text)

        if len(results) == 0:
            print('zero nominatim results')
            output = None
        else:
            if centre:
                output = dict(lat=results[0]['lat'], lon=results[0]['lon'])
            elif bbox:
                output = dict(bbox=results[0]['boundingbox'])
            else:
                if 'osm_id' not in results[0]:
                    print('no osm id found in nominatim results')
                    output = None
                else:
                    osm_id = results[0]['osm_id']
                    osm_type = results[0]['osm_type']
                    output = dict(osm_id=osm_id, osm_type=osm_type)

        self.cache[cache_key] = output
        if len(self.cache) % self.save_frequency == 0:
            self.save_cache()
        return output


def replace_overpass_turbo_shortcuts(query, nominatim, bbox=None):

    query, error = replace_nominatim(query, nominatim)
    if error is not None:
        return query, error

    query = replace_variables(query)

    query, error = replace_date(query)
    if error is not None:
        return query, error

    query = replace_bbox(query, bbox=bbox)

    return query, None


def replace_date(query):
    if '{{date' not in query:
        return query, None

    current_date = date(year=2022, month=7, day=4)
    interval_map = dict(second=1, minute=60, hour=3600, day=86400, week=604800, month=2628000, year=31536000)
    for unit, interval in list(interval_map.items()):  # plural
        interval_map[unit + 's'] = interval

    relative_dates = re.findall(r'{{date:(.*?)}}', query)

    for relative_date in relative_dates:
        try:
            datetime.strptime(relative_date, '%Y-%m-%dT%H:%M:%SZ')
            query = query.replace('{{date:' + relative_date + '}}', relative_date)
            continue
        except Exception:
            pass

        try:
            num_unit = re.match(r'(\d+)\s*(\w+)', relative_date.strip().lower())
            if num_unit is None:
                error = dict(type='relative_date', message='Wrong relative date format')
                return query, error
            num = int(num_unit.group(1))
            unit = num_unit.group(2)

            relative_date_seconds = num * interval_map[unit]
        except KeyError:
            error = dict(type='relative_date', message='KeyError when processing relative date')
            return query, error
        new_date = current_date - timedelta(seconds=relative_date_seconds)
        formatted_new_date = new_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        query = query.replace('{{date:' + relative_date + '}}', formatted_new_date)
    return query, None


def replace_nominatim(query, nominatim):

    def get_replacement(output):

        if output['osm_type'] == 'node':
            return f"node(id:" + str(output["osm_id"]) + ')'

        assert output['osm_type'] in ['way', 'relation']
        add_number = dict(way=2400000000, relation=3600000000)
        r = 'area(id:'
        r += str(output["osm_id"] + add_number[output['osm_type']])
        r += ')'
        return r

    geocode_areas = re.findall(r'{{geocodeArea:(.*?)}}', query)
    for geocode_area in geocode_areas:
        nominatim_output = nominatim.query(geocode_area)

        if nominatim_output is None:
            error = dict(type='nominatim', message='geocodeArea not found by Nominatim', value=geocode_area)
            return query, error
        query = query.replace('{{geocodeArea:' + geocode_area + '}}', get_replacement(nominatim_output))

    nominatim_areas = re.findall(r'{{nominatimArea:(.*?)}}', query)
    for nominatim_area in nominatim_areas:
        nominatim_output = nominatim.query(nominatim_area)

        if nominatim_output is None:
            error = dict(type='nominatim', message='nominatimArea not found by Nominatim', value=nominatim_area)
            return query, error

        query = query.replace('{{nominatimArea:' + nominatim_area + '}}', get_replacement(nominatim_output))

    geocode_ids = re.findall(r'{{geocodeId:(.*?)}}', query)
    for geocode_id in geocode_ids:
        nominatim_output = nominatim.query(geocode_id)

        if nominatim_output is None:
            error = dict(type='nominatim', message='geocodeId not found by Nominatim', value=geocode_id)
            return query, error

        replacement = nominatim_output["osm_type"] + "(id:" + str(nominatim_output["osm_id"]) + ")"
        query = query.replace('{{geocodeId:' + geocode_id + '}}', replacement)


    geocode_coords = re.findall(r'{{geocodeCoords:(.*?)}}', query)
    for geocode_coord in geocode_coords:
        nominatim_output = nominatim.query(geocode_coord, centre=True)

        if nominatim_output is None:
            error = dict(type='nominatim', message='geocodeCoords not found by Nominatim', value=geocode_coord)
            return query, error

        replacement = f'{nominatim_output["lat"]},{nominatim_output["lon"]}'
        query = query.replace('{{geocodeCoords:' + geocode_coord + '}}', replacement)


    geocode_bboxs = re.findall(r'{{geocodeBbox:(.*?)}}', query)
    for geocode_bbox in geocode_bboxs:
        nominatim_output = nominatim.query(geocode_bbox, bbox=True)

        if nominatim_output is None:
            error = dict(type='nominatim', message='geocodeBbox not found by Nominatim', value=geocode_bbox)
            return query, error

        replacement = ','.join(nominatim_output["bbox"])
        query = query.replace('{{geocodeBbox:' + geocode_bbox + '}}', replacement)

    return query, None


def replace_variables(query):
    # Solve user defined variables of the form "{{dach=area(4384343)}}" later in the query used by {{dach}}
    for match in re.findall(r'{{([^}]+?)=(.*?)}}', query):
        key, value = match
        query = query.replace('{{' + key + '}}', value)
        query = query.replace('{{ ' + key + ' }}', value)
        query = query.replace('{{' + key + '=' + value + '}}', ' ')
    return query


def replace_bbox(query, bbox):
    query = query.replace("{{bbox}}", bbox)
    query = query.replace("{{center}}", bbox)
    return query


def replace_overpass_turbo_vars(query):
    query = query.replace("{{GeocodeArea:", "{{geocodeArea:")

    query = query.replace("{{bbox}}", "44.99,-0.99,44.99,-0.99")
    query = query.replace("{{ bbox }}", "44.99,-0.99,44.99,-0.99")
    query = query.replace("{{center}}", "44.88,-0.88,44.88,-0.88")
    query = query.replace("{{ center }}", "44.88,-0.88,44.88,-0.88")

    geocodeArea = dict()
    if "{{geocodeArea:" in query:
        geo_code_pattern = re.compile(r'{{geocodeArea:(.*?)}}')
        replace_id = 3600069990
        while "{{geocodeArea:" in query:
            match = re.search(geo_code_pattern, query)
            if match is None:
                break
            location = match.group(1).strip().rstrip('"').lstrip('"')
            geocodeArea[replace_id] = location
            query = re.sub(geo_code_pattern, f'area({replace_id})', query, 1)
            replace_id += 1

    nominatimArea = dict()
    if "{{nominatimArea:" in query:
        nominatim_area_pattern = re.compile(r'{{nominatimArea:(.*?)}}')
        replace_id = 3600169990
        while "{{nominatimArea:" in query:
            match = re.search(nominatim_area_pattern, query)
            location = match.group(1).strip().rstrip('"').lstrip('"')
            nominatimArea[replace_id] = location
            query = re.sub(nominatim_area_pattern, f'area({replace_id})', query, 1)
            replace_id += 1

    geocodeId = dict()
    if "{{geocodeId:" in query:
        geo_id_pattern = re.compile(r'{{geocodeId:(.*?)}}')
        replace_id = 3600079990
        while "{{geocodeId:" in query:
            match = re.search(geo_id_pattern, query)
            location = match.group(1).strip().rstrip('"').lstrip('"')
            geocodeId[replace_id] = location
            query = re.sub(geo_id_pattern, f'relation({replace_id})', query, 1)
            replace_id += 1

    geocodeBbox = dict()
    if "{{geocodeBbox:" in query:
        geo_bbox_pattern = re.compile(r'{{geocodeBbox:(.*?)}}')
        replace_id = 10
        while "{{geocodeBbox:" in query:
            match = re.search(geo_bbox_pattern, query)
            location = match.group(1).strip().rstrip('"').lstrip('"')
            geocodeBbox[replace_id] = location
            query = re.sub(geo_bbox_pattern, f'{replace_id}.77,-0.88,44.88,-0.88', query, 1)
            replace_id += 1

    geocodeCoords = dict()
    if "{{geocodeCoords:" in query:
        geo_coords_pattern = re.compile(r'{{geocodeCoords:(.*?)}}')
        replace_id = 10
        while "{{geocodeCoords:" in query:
            match = re.search(geo_coords_pattern, query)
            location = match.group(1).strip().rstrip('"').lstrip('"')
            geocodeCoords[replace_id] = location
            query = re.sub(geo_coords_pattern, f'{replace_id}.66,-0.88,44.88,-0.88', query, 1)
            replace_id += 1

    dates = dict()
    if "{{date:" in query:
        date_pattern = re.compile(r'{{date:(.*?)}}')
        replace_id = 1000
        while "{{date:" in query:
            match = re.search(date_pattern, query)
            date = match.group(1).strip().rstrip('"').lstrip('"')
            dates[replace_id] = date
            query = re.sub(date_pattern, f'{replace_id}-00-00T00:00:00Z', query, 1)
            replace_id += 1

    for match in re.findall(r'{{data:(.+?)}}', query):
        query = query.replace('{{data:' + match + '}}', ' ')

    for match in re.findall(r'{{([^}]+?)=(.*?)}}', query):
        key, value = match
        if key == 'bbox':
            query = query.replace('{{' + key + '=' + value + '}}', ' ')
            continue

        query = query.replace('{{' + key + '}}', value)
        query = query.replace('{{ ' + key + ' }}', value)
        query = query.replace('{{' + key + '=' + value + '}}', ' ')

    query = query.strip()
    return query


if __name__ == '__main__':
    with open('./data/paper5/dataset.test.query') as f:
        ref_queries = [line.strip() for line in f]
    with open('./outputs/codet5-small_paper5_test1/evaluation/preds_test_beams4_paper5_test1_codet5_small.txt') as f:
        out_queries = [line.strip() for line in f]

    #converter_url = 'http://petty.cl.uni-heidelberg.de:12346/api/convert'
    converter_url = 'http://overpass-api.de/api/convert'

    converter = OverpassXMLConverter(url=converter_url)
    score = get_oqo_score(out_queries, ref_queries, converter, weights=[1, 1, 1])
    print(score)
