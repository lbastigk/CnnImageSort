# CNN image sorter

This project is based on a Masters Project that uses a json-defined model to generate a python CNN class.

The required modifications are, that the ipynb script is separated into train and sort.

The directory currently is:
```
.
├── categories
│   ├── cat1
│   └── cat2
├── ml-projekt-skater-erkennung.ipynb
├── model.jsonc
├── Requirements.md
├── requirements.txt
├── sort.py
├── suggestions
├── tosort
└── train.py
```
Where train takes images from `categories/<category>/*.{any image file}`.

Sort uses all images in `./tosort/` and places them into `./suggestions/<category>`.

Note that we have to scale all images before using them to train or sort. A simple stretch-rescale should still be good enough.
Of corse, do not modify the images itself!
