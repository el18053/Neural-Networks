# Neural-Networks

# Βαθιά Μάθηση

## Description

This project explores the application of deep learning techniques in **Image Captioning**, a task that combines Computer Vision and Natural Language Processing to generate textual descriptions from images. The work builds upon TensorFlow's tutorial ["Image captioning with visual attention"](https://www.TensorFlow.org/tutorials/text/image_captioning) and extends it by utilizing a custom dataset and implementing various improvements.

## Features

- Utilize the **Flickr30k-images-ecemod** dataset, specifically curated for this project.
- Implement visual attention mechanisms to enhance caption generation.
- Evaluate the quality of generated captions using the BLEU metric.
- Provide an optional in-class competition for testing predictions on unseen data.

## Dataset

The **Flickr30k-images-ecemod** dataset includes:
- A folder `image_dir` containing 31,783 images.
- `captions_new.csv`: A file with 148,915 captions corresponding to the images.
- `train_files.csv`: List of 21,000 images for training.
- `test_files.csv`: List of 4,524 images for testing.

Each image has five captions generated via Amazon Mechanical Turk.

### Data Preparation

1. **Download Images**:
   ```python
   # Download image files
   image_zip = tf.keras.utils.get_file(
       'flickr30k-images-ecemod.zip',
       cache_subdir=os.path.abspath('.'),
       origin='https://spartacus.1337.cx/flickr-mod/flickr30k-images-ecemod.zip',
       extract=True
   )
   os.remove(image_zip)
   ```

2. **Download Captions**:
   ```python
   # Download captions file
   captions_file = tf.keras.utils.get_file(
       'captions_new.csv',
       cache_subdir=os.path.abspath('.'),
       origin='https://spartacus.1337.cx/flickr-mod/captions_new.csv',
       extract=False
   )

   # Download train files list
   train_files_list = tf.keras.utils.get_file(
       'train_files.csv',
       cache_subdir=os.path.abspath('.'),
       origin='https://spartacus.1337.cx/flickr-mod/train_files.csv',
       extract=False
   )

   # Download test files list
   test_files_list = tf.keras.utils.get_file(
       'test_files.csv',
       cache_subdir=os.path.abspath('.'),
       origin='https://spartacus.1337.cx/flickr-mod/test_files.csv',
       extract=False
   )
   ```

3. **Organize Data**:
   ```python
   import collections
   import pathlib

   path = pathlib.Path(".")
   IMAGE_DIR = "image_dir"

   # Load captions
   captions = (path / captions_file).read_text().splitlines()
   captions = (line.split('\t') for line in captions)
   captions = ((fname.split('#')[0], caption) for fname, caption in captions)

   cap_dict = collections.defaultdict(list)
   for fname, cap in captions:
       cap_dict[fname].append(cap)

   # Train files
   train_files = (path / train_files_list).read_text().splitlines()
   train_captions = [(str(path / IMAGE_DIR / fname), cap_dict[fname]) for fname in train_files]

   # Test files
   test_files = (path / test_files_list).read_text().splitlines()
   test_captions = [(str(path / IMAGE_DIR / fname), cap_dict[fname]) for fname in test_files]
   ```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset as described in the **Dataset** section.

## Usage

1. Open the provided Jupyter notebook in Google Colab:
   - Run all setup cells, including dataset download and library imports.
   - Follow the steps to preprocess data, train the model, and evaluate results.

2. Generate captions:
   ```bash
   python generate_captions.py --image_path /path/to/image.jpg
   ```

3. Participate in the optional competition by submitting predictions to the class Codalab platform.

## Evaluation

- The quality of the generated captions is measured using the **BLEU score**.
- Various hyperparameters and techniques such as beam search can be tuned for better results.

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and open a pull request. For significant changes, consider opening an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please contact the course instructors or use the discussion forum available on the course platform.
