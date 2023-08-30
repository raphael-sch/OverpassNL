import copy
import re

import tqdm
import argparse

from evaluation import replace_overpass_turbo_shortcuts
from evaluation import Nominatim, Overpass, OverpassXMLConverter
from evaluation import get_exact_match, get_oqo_score


parser = argparse.ArgumentParser(description='Define evaluation parameter. Execution Accuracy can run for several hours.')
parser.add_argument('--ref_file', default='../dataset/dataset.dev', type=str)
parser.add_argument('--model_output_file', default='./output/dataset.dev.query', type=str)
parser.add_argument('--retry_errors', default=False, type=lambda x: (str(x).lower() in ['true', 'yes']), help='Retry queries where an error was returned.')
parser.add_argument('--compute_oqs', default=True, type=lambda x: (str(x).lower() in ['true', 'yes']))
parser.add_argument('--compute_execution', default=True, type=lambda x: (str(x).lower() in ['true', 'yes']))
parser.add_argument('--overpass_url', default='http://localhost:12346/api/interpreter', type=str)
parser.add_argument('--nominatim_url', default='https://nominatim.openstreetmap.org/search.php', type=str)
parser.add_argument('--convert_url', default='http://localhost:12346/api/convert', type=str)
parser.add_argument('--cache_save_frequency', default=100, type=int)
parser.add_argument('--cache_dir', default='./cache', type=str)
opts = parser.parse_args()

# keep 300 to have consistent evaluation
timeout = 300

cache_dir = opts.cache_dir
split = opts.ref_file.split(".")[-1]
overpass = Overpass(opts.overpass_url,
                    cache_dir=cache_dir,
                    cache_filename=f'overpass_{split}_cache',
                    save_frequency=opts.cache_save_frequency)
nominatim = Nominatim(opts.nominatim_url, cache_dir=cache_dir, save_frequency=opts.cache_save_frequency)
converter = OverpassXMLConverter(opts.convert_url, cache_dir=cache_dir, save_frequency=opts.cache_save_frequency)


def main():
    print('start evaluation')
    ref_nls = read_file(opts.ref_file + '.nl')
    ref_raw_queries = read_file(opts.ref_file + '.query')
    ref_bboxes = read_file(opts.ref_file + '.bbox')
    out_raw_queries = read_file(opts.model_output_file)

    assert len(ref_raw_queries) == len(ref_nls)
    assert len(ref_raw_queries) == len(out_raw_queries)
    assert len(ref_raw_queries) == len(ref_bboxes)

    metrics = dict()

    if opts.compute_oqs:
        oqo, oqo_output_file = run_overpass_query_overlap(out_raw_queries,
                                                          ref_raw_queries,
                                                          weights=[1, 1, 1]
                                                          )
        metrics['kv_overlap'] = oqo['kv_overlap']
        metrics['xml_overlap'] = oqo['xml_overlap']
        metrics['chrf'] = oqo['chrf']
        metrics['oqo'] = oqo['oqo']
    for key, value in metrics.items():
        print(f'{key}: {round(value, 2)}')

    if opts.compute_execution:
        execution_accuracy, execution_soft_accuracy, exec_output_file = run_execution_accuracy(ref_nls, ref_raw_queries, ref_bboxes, out_raw_queries)
        metrics['execution_accuracy'] = execution_accuracy
        metrics['execution_soft_accuracy'] = execution_soft_accuracy

    print('\n\n')
    if opts.compute_oqs:
        print('Overpass Query Similarity results wrote to: ' + oqo_output_file)
    if opts.compute_execution:
        print('Execution results wrote to: ' + exec_output_file)
    for key, value in metrics.items():
        print(f'{key}: {round(value, 2)}')


def run_overpass_query_overlap(out_raw_queries, ref_raw_queries, weights):
    score, scores = get_oqo_score(out_raw_queries, ref_raw_queries, converter, weights)

    exact_match, exact_matches = get_exact_match(out_raw_queries, ref_raw_queries)
    score['exact_match'] = exact_match


    output_filename = 'results_oqs_' + opts.model_output_file.split('/')[-1]
    output_file = '/'.join(opts.model_output_file.split('/')[:-1] + [output_filename])

    with open(output_file, 'w') as f:
        f.write('Model File: ' + opts.model_output_file + '\n')
        f.write('Reference File: ' + opts.ref_file + '.query' + '\n')
        f.write('Summary'.center(100, '-') + '\n')
        f.write(f'exact_match: {round(score["exact_match"], 2)}' + '\n')
        f.write(f'OQS: {round(score["oqo"], 2)}' + '\n')
        f.write(f'kv_overlap: {round(score["kv_overlap"], 2)}' + '\n')
        f.write(f'xml_overlap: {round(score["xml_overlap"], 2)}' + '\n')
        f.write(f'chrf: {round(score["chrf"], 2)}' + '\n')
        f.write(f'Total number of queries: {len(scores["oqo"])}' + '\n')
        f.write('Detailed (OQS, kv_overlap, xml_overlap, chrf, exact_match)'.center(100, '-') + '\n')
        for i in range(len(scores["oqo"])):
            values = [scores[key][i] for key in ['oqo', 'kv_overlap', 'xml_overlap', 'chrf']] + [exact_matches[i]]
            f.write(', '.join([str(round(value, 2)) for value in values]) + '\n')

    print('Overpass Query Overlap results wrote to: ' + output_file)

    return score, output_file


def run_execution_accuracy(ref_nls, ref_raw_queries, ref_bboxes, out_raw_queries):

    results_output = list()
    results_num = list()
    results_execution_acc = list()
    results_execution_soft_acc = list()

    total = 0
    ref_empty = 0
    out_empty = 0
    ref_errors = 0
    out_errors = 0
    ref_error_types = dict(nominatim=0,
                           parse_error=0,
                           timeout=0,
                           static_error_diff=0,
                           static_error_timestamp=0,
                           static_error_re=0,
                           runtime_error_timeout=0,
                           server_error_timeout=0,
                           static_error_convert=0,
                           memory_error=0,
                           static_error_bbox=0,
                           static_error_attribute_value=0,
                           relative_date=0,
                           unknown_error=0)
    out_error_types = copy.copy(ref_error_types)

    for i in tqdm.tqdm(list(range(len(ref_raw_queries)))):
        print('\n\n')
        results_execution_acc.append(0)
        results_execution_soft_acc.append(0)

        ref_nl = ref_nls[i]
        ref_raw_query = ref_raw_queries[i]
        ref_bbox = ref_bboxes[i]
        out_raw_query = out_raw_queries[i]

        ref_query, ref_error = prepare_query(ref_raw_query, bbox=ref_bbox, timeout=timeout)
        out_query, out_error = prepare_query(out_raw_query, bbox=ref_bbox, timeout=timeout)
        if out_error:
            print('nominatim error')
            print(out_error)

        print('Input: ' + ref_nl)
        print('Reference:')
        print(ref_raw_query)
        print('Model:')
        print(out_raw_query)

        total += 1

        # check if error during preprocessing. Don't match if any errors
        if ref_error:
            print('Should not happen')
            ref_errors += 1
            ref_error_types[ref_error['type']] += 1
        if out_error:
            print(f'Error in model query preparation: ' + str(out_error))
            out_errors += 1
            out_error_types[out_error['type']] += 1
            results_output.append(out_error['type'])
            results_num.append(-1)
        if ref_error or out_error:
            continue

        ref_results, ref_num = overpass.query(ref_query)
        if opts.retry_errors and int(ref_num) < 0:
            ref_results, ref_num = overpass.query(ref_query, use_cache=False)
        ref_num = int(ref_num)

        out_results, out_num = overpass.query(out_query)
        if opts.retry_errors and int(out_num) < 0:
            out_results, out_num = overpass.query(out_query, use_cache=False)
        out_num = int(out_num)
        results_output.append(out_results)
        results_num.append(out_num)

        if type(out_results) == str:
            print('Execution Hash/Error: ' + out_results)
        print('Number of reference items: ' + str(ref_num))
        print('Number of model items: ' + str(out_num))

        if '_error' in ref_results:
            print('Should not happen')
            ref_errors += 1
            ref_error_types[ref_results] += 1
        if '_error' in out_results:
            out_errors += 1
            out_error_types[out_results] += 1
        if '_error' in ref_results or '_error' in out_results:
            continue

        # check if empty results. Don't match if both empty
        if ref_num == 0:
            print('Should not happen')
            ref_empty += 1
        if out_num == 0:
            out_empty += 1
        if ref_num == 0 or out_num == 0:
            continue

        if ref_results == out_results:
            print('execution match')
            results_execution_acc[-1] = 1

        # soft match
        if 'count_' in ref_results and 'count_' in out_results:
            #results_execution_soft_acc[-1] = min([ref_num, out_num]) / max([ref_num, out_num])
            results_execution_soft_acc[-1] = int(ref_num == out_num)
        elif int(ref_num) > 0 and int(out_num) > 0:
            num_elements_match = len(set(ref_results) & set(out_results))
            soft_acc = num_elements_match / max([len(set(ref_results)), len(set(out_results))])
            results_execution_soft_acc[-1] = soft_acc
        print('soft exec accuracy: ', results_execution_soft_acc[-1])

    print('overpass.num_cache_hits', overpass.num_cache_hits)
    print('overpass.num_url_requests', overpass.num_url_requests)
    nominatim.save_cache()
    overpass.save_cache()

    assert len(results_execution_acc) == total
    assert len(results_output) == total
    assert len(results_num) == total

    execution_accuracy = sum(results_execution_acc) / total * 100
    execution_soft_accuracy = sum(results_execution_soft_acc) / total * 100

    print('total', total)
    print('')
    print('ref_errors', ref_errors)
    print('out_errors', out_errors)
    print('ref_error_types', ref_error_types)
    print('out_error_types', out_error_types)
    print('')
    print('ref_empty', ref_empty)
    print('out_empty', out_empty)
    print('')
    print('execution accuracy', round(execution_accuracy, 1))
    print('execution soft accuracy', round(execution_soft_accuracy, 1))

    if any(e > 0 for e in ref_error_types.values()):
        print('ERRORS FOUND WHEN EXECUTING THE REFERENCE QUERIES. SHOULD NOT HAPPEN! PLEASE DELETE CACHE AND TRY AGAIN')

    output_filename = 'results_execution_' + opts.model_output_file.split('/')[-1]
    output_file = '/'.join(opts.model_output_file.split('/')[:-1] + [output_filename])

    with open(output_file, 'w') as f:
        f.write('Model File: ' + opts.model_output_file + '\n')
        f.write('Reference File: ' + opts.ref_file + '.query' + '\n')
        f.write('Summary'.center(100, '-') + '\n')
        f.write(f'execution accuracy: {round(execution_accuracy, 2)}' + '\n')
        f.write(f'execution soft accuracy: {round(execution_soft_accuracy, 2)}' + '\n')
        f.write(f'Total number of queries: {total}' + '\n')
        f.write(f'Errors when executing model queries: {out_error_types}' + '\n')
        f.write(f'Number of errors when executing model queries: {sum(out_error_types.values())}' + '\n')
        f.write(f'Errors when executing reference queries: {ref_error_types}' + '\n')
        f.write(f'Number of errors when executing reference queries: {sum(ref_error_types.values())}' + '\n')
        f.write('Detailed (Execution Match, Execution Soft Match, Output Hash, Number of Outputs)'.center(100, '-') + '\n')
        for execution_acc, execution_soft_acc, output_hash, num in zip(results_execution_acc, results_execution_soft_acc, results_output, results_num):
            if not type(output_hash) == str:
                output_hash = 'elements'
            f.write(', '.join([str(execution_acc), str(execution_soft_acc), output_hash, str(num)]) + '\n')

    print('Execution results wrote to: ' + output_file)

    return execution_accuracy, execution_soft_accuracy, output_file


def read_file(path):
    lines = list()
    with open(path) as f:
        for i, line in enumerate(f):
            lines.append(line.strip())
    return lines


def prepare_query(query, timeout=500, bbox=None, replace_shortcuts=True):

    if replace_shortcuts:
        query, error = replace_overpass_turbo_shortcuts(query, nominatim, bbox=bbox)
        if error:
            return query, error

    query = adjust_maxsize(query)
    query = adjust_timeout(query, timeout=timeout)

    query = re.sub("\[out:[^\]]+\]", "[out:json]", query)
    if "[out:json]" not in query:
        query = "[out:json]" + query

    #if '{{' in query and '}}' in query:
    #    print('brackets left in query', query)
        #raise ValueError('brackets left in query')
        #return query, 'brackets_left_error'

    return query, None


def adjust_maxsize(query, maxsize=1073741824):
    maxsize_value = re.search("maxsize:(\d+)", query)
    if maxsize_value:
        cur_maxsize = maxsize_value.group(1)
        maxsize = min(int(cur_maxsize), maxsize)
        query = query.replace("maxsize:" + cur_maxsize, "maxsize:" + str(maxsize))
    return query


def adjust_timeout(query, timeout=500):
    timeout_value = re.search("timeout:(\d+)", query)
    if timeout_value:
        cur_timeout = timeout_value.group(1)
        query = query.replace("timeout:" + cur_timeout, "timeout:" + str(timeout))
    else:
        if query[0] != '[':
            query = ';' + query
        query = f"[timeout:{str(timeout)}]" + query
    return query


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        overpass.save_cache()
        nominatim.save_cache()
        converter.save_cache()
        exit()
