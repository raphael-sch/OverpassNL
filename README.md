Code for the preprint:  [Staniek et al, "Text-to-OverpassQL: A Natural Language Interface for Complex Geodata Querying of OpenStreetMap"](#)

Live Demo: https://overpassnl.schumann.pub/ <br>

# Dataset
The main dataset is found in the following files:
```
dataset.{train,dev,test}.nl
dataset.{train,dev,test}.query
dataset.{train,dev,test}.bbox
```

where *.nl* are the natural language inputs, *.query* are the Overpass queries and *.bbox* is used during evaluation for queries that use the {{bbox}} variable/shortcut. This ensures that they are evaluated in an area where the gold query returns results. <br>

The following files are used to determine the difficulty of evaluation instances (Figure 5 in paper):
```
dataset.{dev,test}.difficulty_{len_nl, len_query, num_results, train_sim_nl train_sim_oqo, xml_components}
```

where *train_sim_oqo* is used to determine the 333 hard instances in *dataset.{dev,test}.hard.{nl,query}*.



# Evaluation
Download the exact OpenStreetMap database we used for the evaluation in the paper [here](https://www.cl.uni-heidelberg.de/~rschuman/files/overpassnl/overpass_clone_db.zip) [306 GB]<br>
Unzip the file (381 GB unziped) such that the folder structure is: *evaluation/overpass_clone_db/db* <br>
Install *docker* and *docker-compose* and start the container with the Overpass API. 

```
cd evaluation/docker
docker-compose up
```

This will start the Overpass API as a docker service and expose it at http://localhost:12346/api <br>
If you see permission or file not found errors for *db/osm3s_v0.7.57_areas* or *db/osm3s_v0.7.57_osm_base* be sure to set correct execution permission to those files <br>
You can test the Overpass API with:
```
curl -g 'http://localhost:12346/api/interpreter?data=[out:json];area[name="London"];out;'
```

If this returns an appropriate json output, you are set for the evaluation.

```
cd evaluation
pip install -r requirements.txt
python run_evaluation.py --ref_file ../dataset/dataset.dev --model_output_file ../models/outputs/byt5-base_com08/evaluation/preds_dev_beams4_com08_byt5_base.txt 
```

This will take around 5 hours depending on how many query results were cached in previous runs. Be sure to change the default arguments in the evaluation script if you use a different port for the Overpass API. <br>
The evaluation results will be written to *results_execution...txt* and *results_oqs...txt* in the same dir as the *model_output_file*.

# OverpassT5
Download the model config and weights [here](https://www.cl.uni-heidelberg.de/~rschuman/files/overpassnl/byt5-base_com08.zip) and place the files into *models/outputs/byt5-base_com08/* <br>
Then run the following commands to generate the output queries.
```
cd evaluation
pip install -r requirements.txt
python inference_t5.py --exp_name com08 --model_name byt5-base --data_dir ../dataset --num_beams 4 --splits dev test 
```


# Finetuning
To finetune your own model use the *train_t5.py* script.
```
cd evaluation
pip install -r requirements.txt
python train_t5.py --exp_name default --data_dir ../dataset --model_name google/byt5-base  --gradient_accumulation_steps 4
python inference_t5.py --exp_name default --model_name byt5-base --data_dir ../dataset --num_beams 4 
```

# Python
Recommended Python version is 3.10 for all scripts.

# References
The demo front-end is a fork of https://github.com/rowheat02/osm-gpt <br>
We thank the https://overpass-turbo.eu/ community and Martin Raifer


# Citation
Please cite the following paper:

```
@article {staniek-2023-overpassnl,
 title = "Text-to-OverpassQL: A Natural Language Interface for Complex Geodata Querying of OpenStreetMap",
 author = "Michael Staniek and Raphael Schumann and Maike Züfle and Stefan Riezler",
 year = "2023",
 publisher = "arXiv",
 eprint = "2309.??" 
}
```