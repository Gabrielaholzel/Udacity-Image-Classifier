# Command Line Arguments
## `get_input_args()`

This is the only function inside this file. With this function, I use `argparse` to retrieve three command line arguments from the user. 

### Expected Outcome
This code will input the three command line arguments from the user.

### Check the code
The `check_command_line_arguments` function within [`check_images.py`][file] will check the code.
Test the following:
* Entering no command line arguments when you run check_image.py from the terminal window. This should result in the default values being printed.
* Entering in values of your choice for the command line arguments when you run check_image.py from the terminal window. This should result in the values you entered being printed.

### Purpose
The purpose of command line arguments is to provide a way for the programs to be more flexible by allowing external inputs (command line arguments) to be input into a program. The key is that these external arguments can change as to allow more flexibility in the program.

#### Usage of Argparse:
The argparse module is used to input the following external inputs into `check_image.py`. Below are the three external inputs the `check_image.py` program will need to retrieve from the user along with the suggested default values each should have.

* Folder that contains the pet images
    _pet_images/_
* The CNN model architecture to use
_resnet_, _alexnet_, or _vgg_ (the latter was chosen as the default). You will find them in `classifier.py`.
* The file that contains the list of valid dognames
_dognames.txt_

The `get_input_args` function creates an argument parser object using [argparse.ArgumentParser][arparse] and then use the [add_argument][add_argument] method to allow the users to enter in these three external inputs from above.

Below is an example of creating an argument parser object and then using add_argument to add an argument that's a path to a folder and a second argument that's an integer.

```python
## Creates Argument Parser object named parser
parser = argparse.ArgumentParser()

## Argument 1: that's a path to a folder
parser.add_argument('--dir', type = str, default = 'pet_images/', 
                    help = 'path to the folder of pet 
```
Below you will find an explanation of the inputs into add_argument.
* `--dir`: The variable name of the argument (here it's `dir`)
* `type`: The type of the argument (here it's a string)
* `default`: The default value (here it's `pet_images/`)
* `help`: The text that will appear if the user types the program name and then `-h` or `--help`. This allows the user to understand what's expected an argument's value.

To access the arguments passed into the program through the argparse object, you will need to use the [parse_args method][arg_parse_method]. The code below demonstrates how to access the arguments through the argparse extending the example above.

To begin, you will need to assign a variable to `parse_args` and then use that variable to access the arguments of your argparse object. If you are creating the argparse object within a function, you will need to _return_ `parse_args` instead of assigning a variable to it. Also, note that the variable `in_args` points to a **collection** of the command line arguments.

This means to access the one we created in the code above, we have to reference the collection variable name `in_args` then specify the command line argument variable name `dir`. For this example, it would be `in_args.dir`, where `in_args` is the collection variable name and `dir` refers to the command line argument variable name. Notice that you need a dot (.) separating the two variable names. The code below shows the assignment of `in_args` to our parser and then accessing the value of `in_args.dir` with the print statement.
```python
## Assigns variable in_args to parse_args()
in_args = parser.parse_args()

## Accesses values of Argument 1 by printing it
print("Argument 1:", in_args.dir)
```

#### Running a Program using command line arguments
To run a program like `check_images.py`, first, open a terminal window in your preferred workspace. Next, type the following and hit enter to run the program (this example - `check_images.py`). Because no command line arguments are specified after the program name, this will use the default command line arguments that have been defined.
```python
python check_images.py 
```

To run a program like `check_images.py` using the command line argument `--dir`, first, open a terminal window within the your preferred workspace. Next, type the following and hit enter to run the program. Notice that all command line arguments are specified after the program name and they are indicated by the `--` that proceeds their variable name with the value following the variable name.
```python
python check_images.py --dir pet_images/
```






[//]: ()
[file]: <https://github.com/Gabrielaholzel/Udacity-Image-Classifier/blob/ae425a0b76c6a2656d0de03ab3804c4ac595e94e/First_Project/get_input_args.py>
[arparse]: <https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser>
[add_argument]: <https://docs.python.org/3/library/argparse.html#adding-arguments>
[arg_parse_method]: <https://docs.python.org/3/library/argparse.html#the-parse-args-method>
