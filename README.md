# Code for GIST: Graph Inverse Style Transfer for Counterfactual Explainability


Execute the experiment from the main the code would be something like this:

```
python main.py config/<dataset>/run_all.json <run_number>
```

The output files for all explainers will be saved under output/{scope_name}/*

To generate csv files with the results use the following command:
```
python src/utils/generate_results.py --source_folder output/<scope_name>/ --output_file <scope_name>.csv
```