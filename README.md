# WEBIR Final

## Requirements:
* Install Magenta
* Install from requirements.txt
* Stores index files to ppr4env_index/ (Or ![download here](https://drive.google.com/file/d/1lhJUTOcCuqM8UBSyr9coFTTl-xAcHMNI/view?fbclid=IwY2xjawKv4yFleHRuA2FlbQIxMABicmlkETF0ekI5VVhyY3VVZ1dBTWtDAR4B5sAlqB0BivRZvBawoy3TfNPKVFcWrt_qR54VndRCuVj0ohILGAiWXleFvg_aem_sQvaME05fxmz9QChgyQgnQ))
* Store query to query_dataset/
* Store lmd_matched dataset to lmd_matched/ (From ![here](https://colinraffel.com/projects/lmd/))
* Store queries and expanded queries to query_dataset/ (Or ![download here](https://drive.google.com/file/d/1BSNNtcqJzDYBTT8PCmIkfBm8QbVE0O9f/view?fbclid=IwY2xjawKv44hleHRuA2FlbQIxMABicmlkETF0ekI5VVhyY3VVZ1dBTWtDAR6r_Ubl043S9EYjR4V3Inwy5uaG9ZQAvMCtha8gq2aoSZZkO8emEnEG85xRuA_aem_wqbvHqre_y6Y2GyeTqS3Bg))

## Structure:
* analyze_result.py: Analyze experiment results and plot charts.
* bm25_toolkit.py: BM25 retriever.
* experiment.py: Run experiments for all expansion methods (no mixture), calculate MAP and stores results to 'duration_analysis_results_{timestamp}.csv'. Also stores query-document scores to "all_retrieval_results_{timestamp}.pkl" for mixture models.
* midi_file_scanner.py, midi_parser.py: Code related to processing MIDI.
* mixture_experiment.py: Use retrieval score(all_retrieval_results.pkl) to run experiment on mixture models, stores result to 'score_fusion_results_{timestamp}.csv'
* ppr4env_main.py, ppr4env_music_retrieval.py: Code related to MIDI indexing and retrieval.
* preprocess_query.py: Generate all expansions of all queries in query_dataset/midi_queries
* query_dataset_generator.py: Generates queries by sampling segments from the dataset.
* query_expansion.py: Expand MIDI using 4 different methods.
