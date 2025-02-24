# Content-Based Movie Recommendation System

## Dataset

This system uses the **Wikipedia Movie Plots** dataset, which contains approximately 34,886 movie plot summaries. 

### Dataset Source
- **Kaggle Link:** [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- **File:** `wiki_movie_plots_deduped.csv`
- **Columns Used:** `Title`, `Plot`

## Setup

### Requirements
Ensure you have Python installed (>=3.7). Install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

### Downloading the Dataset
1. Visit the [Github Link](https://github.com/kiq005/movie-recommendation/blob/master/src/dataset/wiki_movie_plots_deduped.csv) to get the public movie plots dataset.
2. Download the dataset and save `wiki_movie_plots_deduped.csv` in the project directory.
3. To simplify the dataset within 500 rows, use `data_compressing.ipynb` to get the most recent movies.

## Running the Script
To run the recommendation system, use the following example command:
```bash
python recommend.py "I love thrilling action movies set in space, with a comedic twist." "compressed_movie_data.csv"
```

## Example Output
```
Recommended Movies:
                       Title  similarity
12345  Guardians of the Galaxy    0.765432
67890               Spaceballs    0.654321
13579                Star Wars    0.543210
24680              Galaxy Quest    0.532109
11223                 Serenity    0.521098
```

## How It Works
1. **TF-IDF Vectorization**: The script converts both the user query and movie plot descriptions into numerical vectors using the TF-IDF method.
2. **Cosine Similarity Calculation**: The similarity between the query and each movie plot is calculated.
3. **Top N Recommendations**: The top 5 most similar movies are returned based on the highest similarity scores.

## Demo

Here is a [demo](https://brown.zoom.us/rec/share/WfolYRQ_XFNP6BlD7suOY84dLxAcHRI0DSUhMDUDvMf-JGmwes-afD7pazIJ8aVT.okYaD_Yz82qlDnT4?startTime=1740380354000) showing how the recommendation system works.

## Notes
- The dataset should be placed in the same directory as the script.
- Larger datasets may require more processing time.

## License
This project is for interview purposes only. The dataset is publicly available on Github under their usage terms.

