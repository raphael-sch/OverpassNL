import json
import os
import argparse

from transformers import DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch.utils.data import DataLoader
import torch

from utils import read_overpass_split, OverpassDataset
from evaluation import get_exact_match, get_oqo_score, OverpassXMLConverter

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--data_dir', default='../dataset', type=str)
parser.add_argument('--splits', nargs='+', help='list of strings', default=['test', 'dev'])
parser.add_argument('--exp_name', default='e10_lr0004', type=str)
parser.add_argument('--output_dir', default='./outputs/train', type=str)
parser.add_argument('--model_name', default='codet5-small', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_beams', nargs='+', help='list of strings', default=[1, 4, 8])
parser.add_argument('--max_length', default=600, type=int)
parser.add_argument('--converter_url', default="http://localhost:12346/api/convert", type=str)
parser.add_argument('--compute_oqo', default=False, type=lambda x: (str(x).lower() in ['true', 'yes']))
opts = parser.parse_args()
print(opts)

exp_name = opts.exp_name
checkpoint_name = opts.model_name + '_' + opts.exp_name
output_dir = opts.output_dir
print('load model from ' + checkpoint_name)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
converter = OverpassXMLConverter(url=opts.converter_url)

def main():
    model_dir = os.path.join(output_dir, checkpoint_name)
    print('start load tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print('start load model')
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    print('model loaded')
    model.config.max_length = opts.max_length
    model.eval()

    for split in opts.splits:
        print('split', split)
        for beams in opts.num_beams:
            print('beams', beams)
            run_inference(model_dir, model, tokenizer, split, int(beams))


def run_inference(model_dir, model, tokenizer, split, beams):
    texts, labels = read_overpass_split(opts.data_dir + f"/dataset.{split}")
    dataset = OverpassDataset(texts, labels, tokenizer, split, inference_only=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')
    batch_size = max(1, int(opts.batch_size / beams))
    print('batch_size', batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)
    num_batches = len(data_loader)

    outputs = list()
    for batch_id, batch in enumerate(data_loader):
        print(f'batch {batch_id+1}/{num_batches}')

        output = model.generate(batch['input_ids'].to(device),
                                max_new_tokens=opts.max_length,
                                min_length=10,
                                num_beams=beams,
                                do_sample=False,
                                early_stopping=False,
                                repetition_penalty=1.0,
                                num_return_sequences=1
                                )
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        outputs.extend(decoded_output)

    print('compute metric')
    metrics = dict()
    exact_match, _ = get_exact_match(outputs, labels)
    metrics['exact_match'] = round(exact_match, 2)

    if opts.compute_oqo:
        converter.api_calls = 0
        oqo_score, _ = get_oqo_score(outputs, labels, converter=converter)
        print('api calls', converter.api_calls)
        converter.api_calls = 0
        converter.save_cache()
        metrics['chrf'] = round(oqo_score['chrf'], 2)
        metrics['kv_overlap'] = round(oqo_score['kv_overlap'], 2)
        metrics['xml_overlap'] = round(oqo_score['xml_overlap'], 2)
        metrics['oqo'] = round(oqo_score['oqo'], 2)

        print('chrf', metrics['chrf'])
        print('oqo', metrics['oqo'])

    print('exact_match', metrics['exact_match'])
    evaluation_dir = os.path.join(model_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

    output_filename = f'{split}_beams{beams}_{opts.exp_name}_{opts.model_name.replace("-", "_")}.txt'
    pred_out_filename = os.path.join(evaluation_dir, f'preds_{output_filename}')
    with open(pred_out_filename, 'w') as f:
        for output in outputs:
            f.write(output + '\n')

    metrics_out_filename = os.path.join(evaluation_dir, f'metrics_{output_filename}')
    with open(metrics_out_filename, 'w') as f:
        json.dump(metrics, f)

    print('wrote preds to ' + pred_out_filename)
    print('wrote metrics to ' + metrics_out_filename)


if __name__ == '__main__':
    main()
