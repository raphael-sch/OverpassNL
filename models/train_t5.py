import os
import argparse
import shutil

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

from utils import OverpassDataset
from utils import read_overpass_split, setup_seed, get_comment_queries

from evaluation import get_exact_match, get_oqo_score, OverpassXMLConverter

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--data_dir', default='../dataset', type=str)
parser.add_argument('--exp_name', default='e10_lr0004', type=str)
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--model_name', default='Salesforce/codet5-small', type=str)

parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--eval_batch_size', default=64, type=int)
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--max_length', default=600, type=int)  # max_length=600 for byt5?
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--warmup', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.1, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--learning_rate', default=4e-4, type=float)
parser.add_argument('--lr_scheduler_type', default='linear', type=str)
parser.add_argument('--converter_url', default="http://localhost:12346/api/convert", type=str)

parser.add_argument('--use_comments_task', default=False, type=bool)
parser.add_argument('--comment_max_count', default=10, type=int)
parser.add_argument('--comment_min_content_ratio', default=0.80, type=float)
opts = parser.parse_args()
setup_seed(opts.seed)

group = 'train'
print('config', opts)

converter = OverpassXMLConverter(url=opts.converter_url, save_frequency=-1)


def main():
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name)

    model = T5ForConditionalGeneration.from_pretrained(opts.model_name)
    model.config.max_length = opts.max_length

    train_texts, train_labels = read_overpass_split(opts.data_dir + "/dataset.train")
    val_texts, val_labels = read_overpass_split(opts.data_dir + "/dataset.dev")

    comments_dataset = None
    if opts.use_comments_task:
        comments_texts, comments_labels = get_comment_queries(opts.data_dir + '/../dataset/comments.jsonl',
                                                              comment_max_count=opts.comment_max_count,
                                                              comment_min_content_ratio=opts.comment_min_content_ratio)
        comments_dataset = OverpassDataset(comments_texts, comments_labels, tokenizer, 'train')

    train_dataset = OverpassDataset(train_texts, train_labels, tokenizer, 'train', comments_dataset=comments_dataset)
    val_dataset = OverpassDataset(val_texts, val_labels, tokenizer, 'val')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest')

    output_name = f"{opts.model_name.split('/')[1]}_{opts.exp_name}"
    output_path = os.path.join(opts.output_dir, group, output_name)
    args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        eval_steps=50,
        learning_rate=opts.learning_rate,  # maybe 0.0003
        lr_scheduler_type=opts.lr_scheduler_type,
        warmup_ratio=opts.warmup,  # maybe 0.1
        per_device_train_batch_size=int(opts.batch_size / opts.gradient_accumulation_steps),
        gradient_accumulation_steps=opts.gradient_accumulation_steps,
        per_device_eval_batch_size=int(opts.eval_batch_size / opts.gradient_accumulation_steps),
        weight_decay=opts.weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        num_train_epochs=opts.epochs,
        save_strategy="epoch",
        max_grad_norm=opts.max_grad_norm,  # maybe 0.5
        predict_with_generate=True,
        #bf16=True,
        push_to_hub=False,
        overwrite_output_dir=True,
        seed=opts.seed,
        data_seed=opts.seed,
        metric_for_best_model='oqo',
        greater_is_better=True
    )

    compute_metrics = get_compute_metrics_func(tokenizer, val_dataset)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    shutil.rmtree(output_path)
    trainer.save_model(output_dir=output_path)
    trainer.save_state()
    print('model saved to ' + output_path)


def get_compute_metrics_func(tokenizer, val_dataset):
    decoded_labels = val_dataset.labels
    input_texts = val_dataset.texts

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        print('')
        print('preds', preds.shape)
        print('labels', labels.shape)
        print('max_input_length', val_dataset.max_input_length)
        print('max_output_length', val_dataset.max_output_length)
        print('max_total_length', val_dataset.max_total_length)
        print('')

        print('decode pred')
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        assert len(decoded_preds) == len(decoded_labels)

        print('\n')

        n = 4
        for i, (p, l, t) in enumerate(zip(decoded_preds[:n], decoded_labels[:n], input_texts[:n])):

            print('input: ', t)
            print('pred:\n', p)
            print('gold:\n', l)
            print('\n')

        print('compute metric')
        metrics = dict()

        exact_match, _ = get_exact_match(decoded_preds, decoded_labels)
        metrics['exact_match'] = round(exact_match, 2)

        converter.api_calls = 0
        oqo_score, _ = get_oqo_score(decoded_preds, decoded_labels, converter=converter)
        print('api calls', converter.api_calls)
        converter.api_calls = 0
        try:
            converter.save_cache()
        except Exception:
            pass
        metrics['chrf'] = round(oqo_score['chrf'], 2)
        metrics['kv_overlap'] = round(oqo_score['kv_overlap'], 2)
        metrics['xml_overlap'] = round(oqo_score['xml_overlap'], 2)
        metrics['oqo'] = round(oqo_score['oqo'], 2)

        print(metrics)
        return metrics

    return compute_metrics


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        converter.save_cache()
        exit()
