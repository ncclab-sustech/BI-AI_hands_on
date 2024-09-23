# BI-AI_hands_on

## 💳 Credit

### basic python grammer and cnn introduction
PUll from [this tutorial](https://github.com/bbpi2/cnn-pytorch-tutorial/tree/main)

### introducing neural prior into CNN and evaluating similarity
This was heavily borrowed and adapted from [this tutorial](https://github.com/dicarlolab/vonenet/tree/master)
(Based on [Brain-Score](https://github.com/brain-score/vision))



## ✍️ How to Use

Have you used command line/terminal and Anaconda before?

### No Experience

Simply click on the following link to open a [mybinder](https://mybinder.org/) application by clicking the button below:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/bbpi2/cnn-pytorch-tutorial-env/main?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fbbpi2%252Fcnn-pytorch-tutorial%26urlpath%3Dlab%252Ftree%252Fcnn-pytorch-tutorial%252Fnotebooks%252F0_Welcome.ipynb%26branch%3Dmain)

*Note: This may take several minutes to open.*

### Some Experience

*This is recommended since mybinder has limited resources. These instructions were tested for a Windows machine.*

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) by following the prompts in the link, just keep clicking next (it may warn you that you have a space in your directory, this should be fine).
2. Clone this repo to your local directory (`git clone` or download `.zip`).
3. Open `Anaconda Prompt (anaconda3)` from the start menu and move to where you cloned this repo (for me I had to `cd Documents\cnn-pytorch-tutorial-main\cnn-pytorch-tutorial-main`).
4. Create a new virtual conda environment with: `conda create -n cnn-tutorial`
5. Activate this new environment by running: `conda activate cnn-tutorial`. You should now see the following in your terminal:
```bash
(cnn-tutorial) C:>
```
6. Install ipykernel by running: `conda install ipykernel`.
7. Create a Jupyter Kernel and link to your environment by running: `python -m ipykernel install --user --name=cnn-tutorial`.
8. Install key pacakges:
    * update env  `conda env update --file binder/environment.yml`
9. Deactivate the conda environment by running `conda deactivate`. (We should still be in the directory with folders such as `notebook` and `binder` with the same structure as this repo).
10. Start Jupyter Lab by running `jupyter lab` and make sure the kernel set is `cnn-tutorial`.
