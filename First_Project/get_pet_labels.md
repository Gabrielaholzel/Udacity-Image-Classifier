# Creating Pet Image Labels
## Preliminary Knowledge
### How to Read Filenames from a Folder of Files
The folder _pet_images/_ in the workspace contains the 40 images the classifier algorithms was tested on. The filenames of the images in _pet_images/_ identify the animal in each image.

To create the _labels_ for pet images, it is needed to:
* Read all the files' names in the _pet_image/_ folder
* Process the filenames to create the pet image labels
* Format the pet image labels such that they can be matched to:
    * The classifier function labels
    * The dog names in dognames.txt


In the first task, reads the filenames from a folder. To achieve this task the [listdir][listdir] method from the [os python module][os] was imported. The _listdir_ method retrieves all filenames from the files within a folder. These filenames are returned from listdir as a list. The code below demonstrates how to perform this import and retrieval.

```python
## Imports only listdir function from OS module 
from os import listdir  

## Retrieve the filenames from folder pet_images/
filename_list = listdir("pet_images/")

## Print 10 of the filenames from folder pet_images/
print("\nPrints 10 filenames from folder pet_images/")
for idx in range(0, 10, 1):
    print("{:2d} file: {:>25}".format(idx + 1, filename_list[idx]) )
```

### How to Create a Dictionary of Lists (similar to the Results Dictionary)
The Python [Dictionary][dict] is the data structure used for the Pet Image filenames (as **keys**) and a List that contains the filenames associated labels (as **values**). The following are reasons for this data structure choice:
* The key-value pairs of a dictionary are a logical choice because of the need to process the same filenames (keys) with the **classifier function** and compare its returned labels to those of pet image (values)
* Given an input key, retrieval of the associated value is quicker than retrieval from other data structures (e.g. lists).

### Pet Image File Format for Label Matching
Below is a detailed description of the format of the pet image filenames that are used to create the pet image labels.

#### Pet Image Files
The pet image files are located in the folder _pet_images_, in the workspace. Some examples of the filenames you will see are: _Basenji_00963.jpg_, _Boston_terrier_02259.jpg_, _gecko_80.jpg_, _fox_squirrel_01.jpg_.

There are:
* 40 total images pet images
    * 30 images of dogs
    * 10 images of animals that aren't dogs
* Name (label) of image (**needed for comparison**)
    * Contains upper and lower case letters
    * Contains one or more words to describe the image (label)
    * Words are separated by an underscore (_)

### Python Functions to Create Pet Image Labels
The best format for each pet image name, as far as I'm concerned, is:
* Label: with only lower case letters
* Blank space separating each word in a label composed of multiple words
* Whitespace characters stripped from front & end of label

The following string functions are used to achieve the label format:

* [lower()][lower] - places letters in lower case only.
* [split()][split] - returns a list of words from a string, where the words have been separated (split) by the delimiter provided to the split function. If no delimiter is provided, splits on whitespace.
* [strip()][strip] - returns a string with leading & trailing characters removed. If no characters are provided, strips leading & trailing whitespace characters.
* [isalpha()][isalpha] - returns true only when a string contains only alphabetic characters, returns false otherwise.



## Code 
This section is about the function `get_pet_labels` within [`get_pet_labels.py`][file]. This function creates the labels for the pet images, using the filenames of the pet images in the _pet_images_ folder. These images filenames represent the identity of the pet in the image. The pet image labels are considered to represent the "truth" about the classification of the image. The function takes as input the _image_dir_ string, which is the full path to the folder of images that are to be classified, and returns the results dictionary that will contain the pet image filenames and labels. 

Let's start by creating an empty dictionary named _results_dic_.
```python
results_dic = {}
```
Now, let's retrieve the filenames from folder _image_dir_ and save it in a variable called _filename_list_.
```python
    filename_list = listdir(image_dir)
```
Now we add new key-value pairs to dictionary ONLY when key doesn't already exist. For each element inside the _filename_list_ variable, we first check that it doesn't start with a dot (.) due to the workspace's conditions. 
For the appropriate elements, we transform the string to lowercase and separate it by the delimiter underscore (_). This gives us a list comprised of strings and numbers. For example, if the string was `Boston_terrier_02259.jpg`, the list would be `['boston', 'terrier', '02259.jpg']`. 

After that, for every string in the list we check if it only contains alphabetic characters. If this is true, we append the words to a variable called `pet_label`. After this is done for every string, we strip `pet_label` to make sure there are no leading or trailing whitespace characters. 

Finally, we append to the `results_dic`  the element of the `filename_list` as the key, and `pet_label` as the value. 

```python
for i in range(len(filename_list)):
   if filename_list[i][0] != "." and filename_list[i] not in results_dic:
      words_list = filename_list[i].lower().split('_')
      pet_label = ''

      for word in words_list:
          if word.isalpha():
              pet_label += word + ' '
              
      pet_label = pet_label.strip()
      results_dic[filename_list[i]] = [pet_label]
```

### Checking the code
The `check_creating_pet_image_labels` function within [`check_images.py`][check_images] checks the code by printing out the number of key-value pairs and the first 10 key-value pairs.



[//]: ()
[file]: <https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/b16a481fc0b4d54f6ab70ee49400ef18b3535e21/First_Project/get_pet_labels.py>
[check_images]: <https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/b16a481fc0b4d54f6ab70ee49400ef18b3535e21/First_Project/check_images.py>
[listdir]: <https://docs.python.org/3/library/os.html#os.listdir>
[os]: <https://docs.python.org/3/library/os.html>
[dict]: <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>
[lower]: <https://docs.python.org/3/library/stdtypes.html#str.lower>
[split]: <https://docs.python.org/3/library/stdtypes.html#str.split>
[strip]: <https://docs.python.org/3/library/stdtypes.html#str.strip>
[isalpha]: <https://docs.python.org/3/library/stdtypes.html#str.isalpha>
