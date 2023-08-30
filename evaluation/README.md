

OverpassNL - Evaluation
=====================================================

Prepare Overpass API
-------

Download 381 GB clone of OpenStreetMap processed for Overpass and copy to overpass_clone_db folder: LINK

```docker build -t freezed_overpass_api:0.7.57.2 ./```

```docker-compose up```


Run Evaluation
-------
python==3.10

```python run_evaluation.py --ref_file ../dataset/dataset.dev --model_output_file ../model/outputs/byt5-base_com08/evaluation/preds_dev_beams4_com08_byt5_base.txt```


License
-------
Open Data Commons Open Database-Lizenz (ODbL)