# turboSETI-ML
`turboSETI-ML` uses neural networks to detect narrowband drifting signals in filterbank or h5 files. This code was written for **Breakthrough Listen**, the largest scientific program dedicated to the search for life beyond Earth.

## Introduction
The search for aliens is hard, but **Breakthrough Listen** is taking a crack at it.

 One of the primary tools built for this purpose is `turboSETI` (https://github.com/UCBerkeleySETI/turbo_seti), which searches for narrowband drifting signals in frequency-time data gathered by radio telescopes. These signals typically span only a few Hz, but persist longer in time. Signals that originate far from Earth should exhibit a property known as Doppler drift, where the frequency of the signal appears to "drift" over time; the farther the signal source, the larger the drift rate. One example plot of narrowband signals discovered by `turboSETI` can be found below.

![abacad_obs](paper_plots/abacad_observation.png)

This would be an ideal detection, as the signal is present only in the ON observations (first, third, and fifth arrays) when the telescope is pointed _at_ the source, and not present when the telescope is pointed _away_ from the source.

Currently, `turboSETI` is good at finding these signals, but it is not fast. On fine-resolution data from the Green Bank Telescope, where a file contains over 1.74 billion frequency channels to process, `turboSETI` takes approximately 9 hours to run on a 5-minute observation. `turboSETI-ML` aims to perform the same search function as `turboSETI` but at a faster rate.

---

## Dependencies
- python 3
- tensorflow 2.x
- scikit-learn, scikit-image
- numpy, scipy, pandas, matplotlib
- numba (for vast speed optimizations, trust me!)
- astropy
- blimpy (https://github.com/UCBerkeleySETI/blimpy)
- setigen (https://github.com/bbrzycki/setigen)

---

## Usage
There are 3 main steps in the workflow of this code:

1. Generate a training dataset by sampling from a bunch of filterbank/h5 files.
2. Create and train the ML model.
3. Predict using that model on some fil/h5 file.

### Generating the dataset
To create the training set, we need a path to `.fil` or `.h5` files. These files will be split up, and each frequency chunk will be sampled to find parameters for a chi-squared distribution that will be used to generate reasonable noisy backgrounds for training. These parameters will be saved to an `.npz` file.

This can be done with the following code:

```
python3 generate_dataset.py /mnt_blpd12/datax/GC/AGBT19B_999_06/*0000.fil -total 100000 -spf 1000 -fs 17e6 -max_time 3600 --save_name train_params.npz
```

This code takes samples all files matching the pattern `/mnt_blpd12/datax/GC/AGBT19B_999_06/*0000.fil`. It will take a total of 100,000 means, stddevs, and mins from the files, grabbing 1000 samples from each file in the path. The final `.npz` is saved to `train_params.npz`.

**NOTE:** The program will **duplicate** the sampled parameters until there are `-total` samples if one of the following conditions is met:

1. The amount of time the program has taken exceeds `-max_time` seconds(in this case, 3600 seconds/1 hour).
2. The program runs out of files to sample, which occurs when `-spf` or `--samples_per_file` is too small, `-fs` or `--f_shift` is too large, or if there are simply too few files to sample from.

### Training the model

### Prediction
---
## References
`turboSETI`: Enriquez, E., & Price, D. 2019, ascl, ascl (https://github.com/UCBerkeleySETI/turbo_seti)