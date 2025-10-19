# Vietoris-Rips Filtration Visualization

This python package visualizes Rips filtration progression.

## Input
Start with an embedding or point cloud in a pickle file.

CLI args:
--embedding: path to embedding data as a pickle file (.pkl) <br>
--mouse: mouse or experiment ID <br>
--n-samples: number of downsampled samples to include in visualization. This is a major bottleneck source for time, so be careful with this. Default is 1000 <br>
--min-d: minimum diameter/epsilon at the start of the filtration <br>
--max-d: maximim diameter/epsilon at the end of the filtration <br>
--n-steps: number of frames/steps to output in the visualization plot <br>
--sampling-method: ("random", "uniform", "first") downsampling technique <br>

## Output
A plot with the specified arguments. 

# Example usage:
```
poetry run vr_vis --embeddings path/to/embeddings.pkl --mouse C155 --n-samples 1000 --min-d 0.05 --max-d 10.0 --n-steps 10 --sampling-method uniform
```


Author: Emily Ekstrum emily.ekstrum@cuanschutz.edu
